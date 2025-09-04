import { ComponentMap, RolloutOutput } from '../types/index.js';
import { GEPAState } from './state.js';

/**
 * Immutable snapshot of a GEPA run with convenience accessors.
 * Matches Python GEPAResult structure exactly for 1:1 parity.
 */
export class GEPAResult<O = RolloutOutput> {
  // Core data
  candidates: ComponentMap[];
  parents: (number[] | null)[];
  valAggregateScores: number[];
  valSubscores: number[][];
  perValInstanceBestCandidates: Set<number>[];
  discoveryEvalCounts: number[];
  
  // Optional data
  bestOutputsValset?: Array<Array<[number, O[]]>>;
  
  // Run metadata
  totalMetricCalls?: number;
  numFullValEvals?: number;
  runDir?: string;
  seed?: number;
  
  constructor(data: {
    candidates: ComponentMap[];
    parents: (number[] | null)[];
    valAggregateScores: number[];
    valSubscores: number[][];
    perValInstanceBestCandidates: Set<number>[];
    discoveryEvalCounts: number[];
    bestOutputsValset?: Array<Array<[number, O[]]>>;
    totalMetricCalls?: number;
    numFullValEvals?: number;
    runDir?: string;
    seed?: number;
  }) {
    this.candidates = data.candidates;
    this.parents = data.parents;
    this.valAggregateScores = data.valAggregateScores;
    this.valSubscores = data.valSubscores;
    this.perValInstanceBestCandidates = data.perValInstanceBestCandidates;
    this.discoveryEvalCounts = data.discoveryEvalCounts;
    this.bestOutputsValset = data.bestOutputsValset;
    this.totalMetricCalls = data.totalMetricCalls;
    this.numFullValEvals = data.numFullValEvals;
    this.runDir = data.runDir;
    this.seed = data.seed;
  }
  
  // Convenience properties
  get numCandidates(): number {
    return this.candidates.length;
  }
  
  get numValInstances(): number {
    return this.perValInstanceBestCandidates.length;
  }
  
  get bestIdx(): number {
    const scores = this.valAggregateScores;
    return scores.indexOf(Math.max(...scores));
  }
  
  get bestCandidate(): ComponentMap {
    return this.candidates[this.bestIdx];
  }
  
  get bestScore(): number {
    return this.valAggregateScores[this.bestIdx];
  }
  
  get totalEvaluations(): number {
    return this.totalMetricCalls || 0;
  }
  
  get generationsCompleted(): number {
    return this.candidates.length;
  }
  
  // Convenience methods
  nonDominatedIndices(): number[] {
    const nonDominated = new Set<number>();
    
    for (const instanceSet of this.perValInstanceBestCandidates) {
      for (const idx of instanceSet) {
        nonDominated.add(idx);
      }
    }
    
    return Array.from(nonDominated);
  }
  
  lineage(idx: number): number[] {
    const result: number[] = [];
    let current = idx;
    
    while (current !== null && current !== undefined) {
      result.unshift(current);
      const parentList = this.parents[current];
      if (!parentList || parentList.length === 0) {
        break;
      }
      current = parentList[0]; // Take first parent
    }
    
    return result;
  }
  
  bestK(k: number): Array<{ idx: number; candidate: ComponentMap; score: number }> {
    const indexed = this.valAggregateScores.map((score, idx) => ({ idx, score }));
    indexed.sort((a, b) => b.score - a.score);
    
    return indexed.slice(0, k).map(item => ({
      idx: item.idx,
      candidate: this.candidates[item.idx],
      score: item.score
    }));
  }
  
  instanceWinners(instanceIdx: number): Set<number> {
    return this.perValInstanceBestCandidates[instanceIdx] || new Set();
  }
  
  static fromState<O = RolloutOutput>(state: GEPAState): GEPAResult<O> {
    // Map the actual state property names
    return new GEPAResult<O>({
      candidates: [...state.programs],
      parents: state.programParents.map(p => p.length > 0 ? [...p] : null),
      valAggregateScores: [...state.programFullScoresValSet],
      valSubscores: state.programSubScoresValSet.map(scores => [...scores]),
      perValInstanceBestCandidates: state.programAtParetoFrontValset.map(s => new Set(s)),
      discoveryEvalCounts: [...state.numMetricCallsByDiscovery],
      totalMetricCalls: state.totalNumEvals,
      bestOutputsValset: state.trackBestOutputs ? state.bestOutputsValset as any : undefined
    });
  }
  
  toDict(): Record<string, any> {
    const candidates = this.candidates.map(cand => ({ ...cand }));
    
    return {
      candidates,
      parents: this.parents,
      valAggregateScores: this.valAggregateScores,
      valSubscores: this.valSubscores,
      bestOutputsValset: this.bestOutputsValset,
      perValInstanceBestCandidates: this.perValInstanceBestCandidates.map(s => Array.from(s)),
      discoveryEvalCounts: this.discoveryEvalCounts,
      totalMetricCalls: this.totalMetricCalls,
      numFullValEvals: this.numFullValEvals,
      runDir: this.runDir,
      seed: this.seed,
      bestIdx: this.bestIdx,
      bestScore: this.bestScore,
      bestCandidate: this.bestCandidate
    };
  }
  
  toJSON(): object {
    return this.toDict();
  }
}