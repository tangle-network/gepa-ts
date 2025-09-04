import { ComponentMap, RolloutOutput } from '../types/index.js';
import * as fs from 'fs';
import * as path from 'path';

export interface ProgramTrace {
  i: number;
  newProgramIdx?: number;
  [key: string]: any;
}

export class GEPAState {
  i: number = 0;
  totalNumEvals: number = 0;
  numFullDsEvals: number = 0;
  
  // Program storage
  programs: ComponentMap[] = [];
  programParents: number[][] = [];
  programFullScoresValSet: number[] = [];
  programSubScoresValSet: number[][] = [];
  programOutputsValSet: RolloutOutput[][] = [];
  numMetricCallsByDiscovery: number[] = [];
  
  // CRITICAL FIX: Pareto tracking is PER VALIDATION INSTANCE, not global!
  // For each validation example, we track:
  // - The best score seen so far for that example
  // - Which program indices achieve that best score
  paretoFrontValset: number[] = []; // Best score per validation instance
  programAtParetoFrontValset: Set<number>[] = []; // Program indices at pareto front per instance
  
  fullProgramTrace: ProgramTrace[] = [];
  trackBestOutputs: boolean;
  bestOutputsValset?: Array<Array<[number, RolloutOutput]>> = undefined;
  
  constructor(trackBestOutputs: boolean = false) {
    this.trackBestOutputs = trackBestOutputs;
  }
  
  addProgram(
    program: ComponentMap,
    parentProgramIdx: number[],
    valsetScore: number,
    valsetOutputs: RolloutOutput[],
    valsetSubscores: number[],
    numMetricCallsByDiscovery: number
  ): [number, number] {
    const newIdx = this.programs.length;
    
    // Store program data
    this.programs.push(program);
    this.programParents.push(parentProgramIdx);
    this.programFullScoresValSet.push(valsetScore);
    this.programSubScoresValSet.push(valsetSubscores);
    this.numMetricCallsByDiscovery.push(numMetricCallsByDiscovery);
    
    if (this.trackBestOutputs && this.programOutputsValSet) {
      this.programOutputsValSet.push(valsetOutputs);
    }
    
    // Initialize Pareto front if this is the first program
    if (this.paretoFrontValset.length === 0) {
      this.paretoFrontValset = [...valsetSubscores];
      this.programAtParetoFrontValset = valsetSubscores.map(() => new Set([newIdx]));
      
      if (this.trackBestOutputs) {
        this.bestOutputsValset = valsetOutputs.map((output, idx) => 
          [[newIdx, output]]
        );
      }
    } else {
      // Update Pareto front for each validation instance
      for (let taskIdx = 0; taskIdx < valsetSubscores.length; taskIdx++) {
        const oldScore = this.paretoFrontValset[taskIdx];
        const newScore = valsetSubscores[taskIdx];
        
        if (newScore > oldScore) {
          // New best score for this instance
          this.paretoFrontValset[taskIdx] = newScore;
          this.programAtParetoFrontValset[taskIdx] = new Set([newIdx]);
          
          if (this.trackBestOutputs && this.bestOutputsValset) {
            this.bestOutputsValset[taskIdx] = [[newIdx, valsetOutputs[taskIdx]]];
          }
        } else if (newScore === oldScore) {
          // Tie with current best - add to pareto set
          this.programAtParetoFrontValset[taskIdx].add(newIdx);
          
          if (this.trackBestOutputs && this.bestOutputsValset) {
            this.bestOutputsValset[taskIdx].push([newIdx, valsetOutputs[taskIdx]]);
          }
        }
      }
    }
    
    // Find program with best average score (linear pareto front)
    const linearParetoIdx = this.programFullScoresValSet.indexOf(
      Math.max(...this.programFullScoresValSet)
    );
    
    return [newIdx, linearParetoIdx];
  }
  
  updateStateWithNewProgram(
    parentProgramIdx: number[],
    newProgram: ComponentMap,
    valsetScore: number,
    valsetOutputs: RolloutOutput[],
    valsetSubscores: number[],
    runDir: string | null,
    numMetricCallsByDiscoveryOfNewProgram: number
  ): [number, number] {
    return this.addProgram(
      newProgram,
      parentProgramIdx,
      valsetScore,
      valsetOutputs,
      valsetSubscores,
      numMetricCallsByDiscoveryOfNewProgram
    );
  }
  
  /**
   * Get all programs that are on the Pareto front for at least one validation instance.
   * This is used by merge proposer to select candidates.
   */
  getParetoFrontPrograms(): ComponentMap[] {
    const paretoIndices = new Set<number>();
    
    // Collect all program indices that appear in any pareto front
    for (const instanceParetoSet of this.programAtParetoFrontValset) {
      for (const idx of instanceParetoSet) {
        paretoIndices.add(idx);
      }
    }
    
    // Return the actual programs
    return Array.from(paretoIndices).map(idx => this.programs[idx]);
  }
  
  /**
   * Check if a program dominates another on any validation instance
   */
  isDominated(candidateIdx: number, byIdx: number): boolean {
    const candidateScores = this.programSubScoresValSet[candidateIdx];
    const dominatorScores = this.programSubScoresValSet[byIdx];
    
    // For domination, the dominator must be >= on all objectives
    // and strictly > on at least one
    let hasStrictlyBetter = false;
    
    for (let i = 0; i < candidateScores.length; i++) {
      if (dominatorScores[i] < candidateScores[i]) {
        return false; // Candidate is better on this instance
      }
      if (dominatorScores[i] > candidateScores[i]) {
        hasStrictlyBetter = true;
      }
    }
    
    return hasStrictlyBetter;
  }
  
  isConsistent(): boolean {
    const n = this.programs.length;
    const valsetSize = this.paretoFrontValset.length;
    
    return (
      this.programParents.length === n &&
      this.programFullScoresValSet.length === n &&
      this.programSubScoresValSet.length === n &&
      this.programAtParetoFrontValset.length === valsetSize &&
      (!this.trackBestOutputs || !this.programOutputsValSet || this.programOutputsValSet.length === n)
    );
  }
  
  save(runDir: string | null): void {
    if (!runDir) return;
    
    const stateData = {
      i: this.i,
      totalNumEvals: this.totalNumEvals,
      numFullDsEvals: this.numFullDsEvals,
      programs: this.programs,
      programParents: this.programParents,
      programFullScoresValSet: this.programFullScoresValSet,
      programSubScoresValSet: this.programSubScoresValSet,
      numMetricCallsByDiscovery: this.numMetricCallsByDiscovery,
      paretoFrontValset: this.paretoFrontValset,
      // Convert Sets to Arrays for JSON serialization
      programAtParetoFrontValset: this.programAtParetoFrontValset.map(s => Array.from(s)),
      fullProgramTrace: this.fullProgramTrace,
      trackBestOutputs: this.trackBestOutputs,
      programOutputsValset: this.trackBestOutputs ? this.programOutputsValset : undefined,
      bestOutputsValset: this.trackBestOutputs ? this.bestOutputsValset : undefined
    };
    
    fs.mkdirSync(runDir, { recursive: true });
    fs.writeFileSync(
      path.join(runDir, 'state.json'),
      JSON.stringify(stateData, null, 2)
    );
  }
  
  static load(runDir: string): GEPAState {
    const statePath = path.join(runDir, 'state.json');
    const stateData = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
    
    const state = new GEPAState(stateData.trackBestOutputs);
    Object.assign(state, stateData, {
      // Convert Arrays back to Sets
      programAtParetoFrontValset: stateData.programAtParetoFrontValset.map(
        (arr: number[]) => new Set(arr)
      )
    });
    
    return state;
  }
}

export function initializeGEPAState(
  runDir: string | null,
  seedCandidate: ComponentMap,
  valsetEvaluator: (prog: ComponentMap) => Promise<[RolloutOutput[], number[]]> | [RolloutOutput[], number[]],
  trackBestOutputs: boolean = false
): GEPAState | Promise<GEPAState> {
  // Check if we should load existing state
  if (runDir && fs.existsSync(path.join(runDir, 'state.json'))) {
    return GEPAState.load(runDir);
  }
  
  const state = new GEPAState(trackBestOutputs);
  
  const evalResult = valsetEvaluator(seedCandidate);
  
  if (evalResult instanceof Promise) {
    return evalResult.then(([outputs, scores]) => {
      const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      state.addProgram(seedCandidate, [], avgScore, outputs, scores, scores.length);
      state.fullProgramTrace.push({ i: 0, newProgramIdx: 0 });
      return state;
    });
  } else {
    const [outputs, scores] = evalResult;
    const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
    state.addProgram(seedCandidate, [], avgScore, outputs, scores, scores.length);
    state.fullProgramTrace.push({ i: 0, newProgramIdx: 0 });
    return state;
  }
}