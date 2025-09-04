import { GEPAState } from '../core/state.js';
import { ComponentMap } from '../types/index.js';

export interface CandidateSelector {
  select(state: GEPAState): [ComponentMap, number[]];
}

export class CurrentBestCandidateSelector implements CandidateSelector {
  select(state: GEPAState): [ComponentMap, number[]] {
    const bestIdx = state.programFullScoresValSet.indexOf(
      Math.max(...state.programFullScoresValSet)
    );
    
    return [state.programs[bestIdx], [bestIdx]];
  }
}

export class ParetoCandidateSelector implements CandidateSelector {
  private rng: () => number;
  
  constructor(rng?: () => number) {
    this.rng = rng || Math.random;
  }
  
  select(state: GEPAState): [ComponentMap, number[]] {
    // Collect all program indices that are on the Pareto front for ANY instance
    const paretoIndices = new Set<number>();
    
    for (const instanceParetoSet of state.programAtParetoFrontValset) {
      for (const idx of instanceParetoSet) {
        paretoIndices.add(idx);
      }
    }
    
    if (paretoIndices.size === 0) {
      // Fallback to best if no Pareto programs
      return new CurrentBestCandidateSelector().select(state);
    }
    
    // Convert to array for random selection
    const paretoArray = Array.from(paretoIndices);
    
    // Randomly select from Pareto front programs
    const randomIdx = Math.floor(this.rng() * paretoArray.length);
    const selectedProgramIdx = paretoArray[randomIdx];
    
    return [state.programs[selectedProgramIdx], [selectedProgramIdx]];
  }
}

export class RandomCandidateSelector implements CandidateSelector {
  private rng: () => number;
  
  constructor(rng?: () => number) {
    this.rng = rng || Math.random;
  }
  
  select(state: GEPAState): [ComponentMap, number[]] {
    const randomIdx = Math.floor(this.rng() * state.programs.length);
    return [state.programs[randomIdx], [randomIdx]];
  }
}