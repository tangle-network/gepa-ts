import { 
  ComponentMap, 
  DataInst, 
  LoggerProtocol,
  ProposalResult,
  RolloutOutput
} from '../types/index.js';
import { GEPAState } from '../core/state.js';

interface MergeTriple {
  id1: number;
  id2: number;
  ancestor: number;
}

interface MergeResult {
  success: boolean;
  program?: ComponentMap;
  id1?: number;
  id2?: number;
  ancestor?: number;
}

export interface MergeProposerConfig<D = DataInst> {
  logger: LoggerProtocol;
  valset: D[];
  evaluator: (inputs: D[], prog: ComponentMap) => Promise<[RolloutOutput[], number[]]> | [RolloutOutput[], number[]];
  useMerge: boolean;
  maxMergeInvocations: number;
  rng?: () => number;
}

export class MergeProposer<D = DataInst> {
  private config: MergeProposerConfig<D>;
  lastIterFoundNewProgram: boolean = false;
  mergesDue: number = 0;
  totalMergesTested: number = 0;
  private mergesPerformed: [Set<string>, Array<[number, number, string]>] = [new Set(), []];
  private rng: () => number;
  private randomGen: {
    random: () => number;
    choice: <T>(arr: T[]) => T;
    sample: <T>(arr: T[], n: number) => T[];
    choices: <T>(arr: T[], weights: number[], k: number) => T[];
  };
  
  constructor(config: MergeProposerConfig<D>) {
    this.config = config;
    this.rng = config.rng || Math.random;
    
    // Create Python-like random utilities
    this.randomGen = {
      random: this.rng,
      choice: <T>(arr: T[]): T => {
        return arr[Math.floor(this.rng() * arr.length)];
      },
      sample: <T>(arr: T[], n: number): T[] => {
        const shuffled = [...arr];
        for (let i = shuffled.length - 1; i > 0; i--) {
          const j = Math.floor(this.rng() * (i + 1));
          [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled.slice(0, n);
      },
      choices: <T>(arr: T[], weights: number[], k: number): T[] => {
        const result: T[] = [];
        for (let i = 0; i < k; i++) {
          const totalWeight = weights.reduce((a, b) => a + b, 0);
          let random = this.rng() * totalWeight;
          for (let j = 0; j < arr.length; j++) {
            random -= weights[j];
            if (random <= 0) {
              result.push(arr[j]);
              break;
            }
          }
        }
        return result;
      }
    };
  }
  
  get useMerge(): boolean {
    return this.config.useMerge;
  }
  
  get maxMergeInvocations(): number {
    return this.config.maxMergeInvocations;
  }
  
  private doesTripletHaveDesirablePredictors(
    programs: ComponentMap[],
    ancestor: number,
    id1: number,
    id2: number
  ): boolean {
    const foundPredictors: Array<[string, number]> = [];
    const predNames = Object.keys(programs[ancestor]);
    
    for (const predName of predNames) {
      const predAnc = programs[ancestor][predName];
      const predId1 = programs[id1][predName];
      const predId2 = programs[id2][predName];
      
      if ((predAnc === predId1 || predAnc === predId2) && predId1 !== predId2) {
        // We have a predictor that is the same as one of its ancestors
        const sameAsAncestorId = predAnc === predId1 ? 1 : 2;
        foundPredictors.push([predName, sameAsAncestorId]);
      }
    }
    
    return foundPredictors.length > 0;
  }
  
  private filterAncestors(
    i: number,
    j: number,
    commonAncestors: number[],
    aggScores: number[],
    programs: ComponentMap[]
  ): number[] {
    const filtered: number[] = [];
    
    for (const ancestor of commonAncestors) {
      // Check if this triple has been merged
      const tripleKey = `${i},${j},${ancestor}`;
      if (this.mergesPerformed[0].has(tripleKey)) {
        continue;
      }
      
      // Ancestor should not be better than its descendants
      if (aggScores[ancestor] > aggScores[i] || aggScores[ancestor] > aggScores[j]) {
        continue;
      }
      
      // Check if triplet has desirable predictors
      if (!this.doesTripletHaveDesirablePredictors(programs, ancestor, i, j)) {
        continue;
      }
      
      filtered.push(ancestor);
    }
    
    return filtered;
  }
  
  private getAncestors(node: number, parentList: number[][], visited?: Set<number>): Set<number> {
    visited = visited || new Set();
    
    if (visited.has(node)) {
      return visited;
    }
    
    visited.add(node);
    const parents = parentList[node] || [];
    
    for (const parent of parents) {
      if (parent !== null && parent !== undefined && !visited.has(parent)) {
        this.getAncestors(parent, parentList, visited);
      }
    }
    
    return visited;
  }
  
  private findCommonAncestorPair(
    parentList: number[][],
    programIndexes: number[],
    aggScores: number[],
    programs: ComponentMap[],
    maxAttempts: number = 10
  ): MergeTriple | null {
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      if (programIndexes.length < 2) {
        return null;
      }
      
      const [i, j] = this.randomGen.sample(programIndexes, 2);
      if (i === j) {
        continue;
      }
      
      // Ensure i < j
      const [minIdx, maxIdx] = i < j ? [i, j] : [j, i];
      
      const ancestorsI = this.getAncestors(minIdx, parentList);
      const ancestorsJ = this.getAncestors(maxIdx, parentList);
      
      // If one is an ancestor of the other, cannot merge
      if (ancestorsJ.has(minIdx) || ancestorsI.has(maxIdx)) {
        continue;
      }
      
      // Find common ancestors
      const commonAncestors: number[] = [];
      for (const ancestor of ancestorsI) {
        if (ancestorsJ.has(ancestor)) {
          commonAncestors.push(ancestor);
        }
      }
      
      // Filter ancestors
      const filtered = this.filterAncestors(
        minIdx, 
        maxIdx, 
        commonAncestors, 
        aggScores, 
        programs
      );
      
      if (filtered.length > 0) {
        // Select weighted random ancestor
        const weights = filtered.map(a => aggScores[a]);
        const selected = this.randomGen.choices(filtered, weights, 1)[0];
        return { id1: minIdx, id2: maxIdx, ancestor: selected };
      }
    }
    
    return null;
  }
  
  private sampleAndAttemptMergeByCommonPredictors(
    aggScores: number[],
    mergeCandidates: number[],
    programs: ComponentMap[],
    parentList: number[][],
    maxAttempts: number = 10
  ): MergeResult {
    if (mergeCandidates.length < 2 || parentList.length < 3) {
      return { success: false };
    }
    
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const idsToMerge = this.findCommonAncestorPair(
        parentList,
        mergeCandidates,
        aggScores,
        programs,
        10
      );
      
      if (!idsToMerge) {
        continue;
      }
      
      const { id1, id2, ancestor } = idsToMerge;
      
      // Create new program from ancestor
      const newProgram: ComponentMap = { ...programs[ancestor] };
      let progDesc = '';
      
      const predNames = Object.keys(programs[ancestor]);
      
      // Verify all programs have same predictors
      const id1Keys = new Set(Object.keys(programs[id1]));
      const id2Keys = new Set(Object.keys(programs[id2]));
      for (const key of predNames) {
        if (!id1Keys.has(key) || !id2Keys.has(key)) {
          continue; // Skip if predictor missing
        }
      }
      
      for (const predName of predNames) {
        const predAnc = programs[ancestor][predName];
        const predId1 = programs[id1][predName];
        const predId2 = programs[id2][predName];
        
        if ((predAnc === predId1 || predAnc === predId2) && predId1 !== predId2) {
          // One is same as ancestor, use the other
          const sameAsAncestorId = predAnc === predId1 ? 1 : 2;
          newProgram[predName] = sameAsAncestorId === 1 ? predId2 : predId1;
          progDesc += `${sameAsAncestorId === 1 ? id2 : id1},`;
        } else if (predAnc !== predId1 && predAnc !== predId2) {
          // Both different from ancestor, choose based on scores
          let progToGetFrom: number;
          if (aggScores[id1] > aggScores[id2]) {
            progToGetFrom = id1;
          } else if (aggScores[id2] > aggScores[id1]) {
            progToGetFrom = id2;
          } else {
            progToGetFrom = this.randomGen.choice([id1, id2]);
          }
          newProgram[predName] = programs[progToGetFrom][predName];
          progDesc += `${progToGetFrom},`;
        } else if (predId1 === predId2) {
          // Both same, use either
          newProgram[predName] = predId1;
          progDesc += `${id1},`;
        }
      }
      
      // Check if this specific merge was already performed
      const mergeKey = `${id1},${id2},${progDesc}`;
      const found = this.mergesPerformed[1].some(
        ([i1, i2, desc]) => i1 === id1 && i2 === id2 && desc === progDesc
      );
      
      if (found) {
        continue;
      }
      
      // Record the merge
      this.mergesPerformed[1].push([id1, id2, progDesc]);
      this.mergesPerformed[0].add(`${id1},${id2},${ancestor}`);
      
      return {
        success: true,
        program: newProgram,
        id1,
        id2,
        ancestor
      };
    }
    
    return { success: false };
  }
  
  private selectEvalSubsampleForMergedProgram(
    id1: number,
    id2: number,
    state: GEPAState
  ): number[] {
    // Select validation instances where parent programs excel
    const subscores1 = state.programSubScoresValSet[id1];
    const subscores2 = state.programSubScoresValSet[id2];
    
    const indices: number[] = [];
    
    // Find instances where each parent is best
    for (let i = 0; i < subscores1.length; i++) {
      if (subscores1[i] > subscores2[i] || subscores2[i] > subscores1[i]) {
        indices.push(i);
      }
    }
    
    // Limit to reasonable subsample size
    if (indices.length > 5) {
      // Randomly sample 5 indices
      const shuffled = [...indices];
      for (let i = shuffled.length - 1; i > 0; i--) {
        const j = Math.floor(this.rng() * (i + 1));
        [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
      }
      return shuffled.slice(0, 5);
    }
    
    return indices.length > 0 ? indices : [0, 1, 2, 3, 4].slice(0, Math.min(5, subscores1.length));
  }
  
  private findDominatorPrograms(state: GEPAState): number[] {
    // Find programs that are on Pareto front for multiple instances
    const dominatorCounts = new Map<number, number>();
    
    for (const instancePareto of state.programAtParetoFrontValset) {
      for (const progIdx of instancePareto) {
        dominatorCounts.set(progIdx, (dominatorCounts.get(progIdx) || 0) + 1);
      }
    }
    
    // Sort by dominance count
    const sorted = Array.from(dominatorCounts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([idx]) => idx);
    
    return sorted;
  }
  
  async propose(state: GEPAState): Promise<ProposalResult | null> {
    const i = state.i + 1;
    
    // Find merge candidates from Pareto dominators
    const dominators = this.findDominatorPrograms(state);
    
    if (dominators.length < 2) {
      this.config.logger.log(`Iteration ${i}: Not enough dominator programs for merge`);
      return null;
    }
    
    // Attempt merge
    const mergeResult = this.sampleAndAttemptMergeByCommonPredictors(
      state.programFullScoresValSet,
      dominators,
      state.programs,
      state.programParents,
      10
    );
    
    if (!mergeResult.success || !mergeResult.program) {
      this.config.logger.log(`Iteration ${i}: Could not find valid merge candidates`);
      return null;
    }
    
    const { program: mergedProgram, id1, id2 } = mergeResult;
    
    // Select evaluation subsample
    const evalIndices = this.selectEvalSubsampleForMergedProgram(id1!, id2!, state);
    const subsample = evalIndices.map(idx => this.config.valset[idx]);
    
    // Evaluate parents and merged program on subsample
    const eval1Result = this.config.evaluator(subsample, state.programs[id1!]);
    const eval2Result = this.config.evaluator(subsample, state.programs[id2!]);
    const evalMergedResult = this.config.evaluator(subsample, mergedProgram);
    
    const [_, scores1] = eval1Result instanceof Promise ? await eval1Result : eval1Result;
    const [__, scores2] = eval2Result instanceof Promise ? await eval2Result : eval2Result;
    const [___, scoresMerged] = evalMergedResult instanceof Promise ? 
      await evalMergedResult : evalMergedResult;
    
    const sum1 = scores1.reduce((a, b) => a + b, 0);
    const sum2 = scores2.reduce((a, b) => a + b, 0);
    const sumMerged = scoresMerged.reduce((a, b) => a + b, 0);
    
    this.config.logger.log(
      `Iteration ${i}: Merge candidate scores - Parent1: ${sum1.toFixed(2)}, ` +
      `Parent2: ${sum2.toFixed(2)}, Merged: ${sumMerged.toFixed(2)}`
    );
    
    return {
      candidate: mergedProgram,
      tag: 'merge',
      parentProgramIds: [id1!, id2!],
      subsampleScoresBefore: [sum1, sum2],
      subsampleScoresAfter: scoresMerged
    };
  }
  
  scheduleIfNeeded(): void {
    if (this.useMerge && this.totalMergesTested < this.maxMergeInvocations) {
      this.mergesDue += 1;
    }
  }
}