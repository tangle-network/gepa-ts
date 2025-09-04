/**
 * Tournament selection strategy for genetic algorithms
 * Selects candidates through tournament competition
 */

import { CandidateSelector } from './candidate-selector.js';
import { GEPAState } from '../core/state.js';
import { ComponentMap } from '../types/index.js';

export interface TournamentSelectorConfig {
  tournamentSize: number;
  selectionPressure?: number; // Probability of selecting the best in tournament
  replacement?: boolean; // Whether to allow replacement in tournament
  fitnessFunction?: 'validation' | 'training' | 'combined';
}

/**
 * Tournament-based candidate selection
 * Implements tournament selection commonly used in genetic algorithms
 */
export class TournamentCandidateSelector implements CandidateSelector {
  private config: Required<TournamentSelectorConfig>;
  
  constructor(config: TournamentSelectorConfig) {
    this.config = {
      tournamentSize: config.tournamentSize,
      selectionPressure: config.selectionPressure ?? 0.8,
      replacement: config.replacement ?? false,
      fitnessFunction: config.fitnessFunction ?? 'validation'
    };
    
    if (this.config.tournamentSize < 2) {
      throw new Error('Tournament size must be at least 2');
    }
    
    if (this.config.selectionPressure < 0 || this.config.selectionPressure > 1) {
      throw new Error('Selection pressure must be between 0 and 1');
    }
  }
  
  /**
   * Select candidates using tournament selection
   */
  selectCandidates(
    state: GEPAState,
    componentsToUpdate: string[],
    numCandidates: number = 1
  ): ComponentMap[] {
    const selected: ComponentMap[] = [];
    const allPrograms = state.getAllPrograms();
    
    if (allPrograms.length === 0) {
      throw new Error('No programs available for tournament selection');
    }
    
    // Ensure tournament size doesn't exceed population
    const actualTournamentSize = Math.min(this.config.tournamentSize, allPrograms.length);
    
    for (let i = 0; i < numCandidates; i++) {
      const winner = this.runTournament(state, actualTournamentSize);
      selected.push(winner);
    }
    
    return selected;
  }
  
  /**
   * Run a single tournament
   */
  private runTournament(state: GEPAState, tournamentSize: number): ComponentMap {
    const allPrograms = state.getAllPrograms();
    const participants: { program: ComponentMap; score: number; index: number }[] = [];
    const usedIndices = new Set<number>();
    
    // Select tournament participants
    for (let i = 0; i < tournamentSize; i++) {
      let index: number;
      
      if (this.config.replacement) {
        // With replacement - can select same individual multiple times
        index = Math.floor(Math.random() * allPrograms.length);
      } else {
        // Without replacement - each individual selected at most once
        do {
          index = Math.floor(Math.random() * allPrograms.length);
        } while (usedIndices.has(index) && usedIndices.size < allPrograms.length);
        usedIndices.add(index);
      }
      
      const program = allPrograms[index];
      const score = this.getFitness(state, index);
      
      participants.push({ program, score, index });
    }
    
    // Sort participants by fitness (descending)
    participants.sort((a, b) => b.score - a.score);
    
    // Select winner based on selection pressure
    const rand = Math.random();
    let selectedIndex = 0;
    
    if (rand < this.config.selectionPressure) {
      // Select the best
      selectedIndex = 0;
    } else {
      // Probabilistic selection among remaining participants
      // Higher ranked individuals have higher probability
      const remainingProb = 1 - this.config.selectionPressure;
      let cumProb = 0;
      
      for (let i = 1; i < participants.length; i++) {
        // Linear ranking probability
        const rankProb = (participants.length - i) / 
                        ((participants.length * (participants.length - 1)) / 2);
        cumProb += rankProb * remainingProb;
        
        if (rand < this.config.selectionPressure + cumProb) {
          selectedIndex = i;
          break;
        }
      }
    }
    
    return participants[selectedIndex].program;
  }
  
  /**
   * Get fitness score for a program
   */
  private getFitness(state: GEPAState, programIndex: number): number {
    switch (this.config.fitnessFunction) {
      case 'validation':
        return state.getAverageValScore(programIndex);
        
      case 'training':
        return state.getAverageTrainScore(programIndex);
        
      case 'combined':
        const trainScore = state.getAverageTrainScore(programIndex);
        const valScore = state.getAverageValScore(programIndex);
        // Weighted average favoring validation
        return 0.3 * trainScore + 0.7 * valScore;
        
      default:
        return state.getAverageValScore(programIndex);
    }
  }
  
  /**
   * Select candidates for breeding (genetic crossover)
   */
  selectForBreeding(
    state: GEPAState,
    numPairs: number = 1
  ): Array<[ComponentMap, ComponentMap]> {
    const pairs: Array<[ComponentMap, ComponentMap]> = [];
    
    for (let i = 0; i < numPairs; i++) {
      // Run two tournaments to select parents
      const parent1 = this.runTournament(state, this.config.tournamentSize);
      const parent2 = this.runTournament(state, this.config.tournamentSize);
      
      // Ensure parents are different
      if (JSON.stringify(parent1) !== JSON.stringify(parent2)) {
        pairs.push([parent1, parent2]);
      } else {
        // If same, try once more
        const altParent2 = this.runTournament(state, this.config.tournamentSize);
        pairs.push([parent1, altParent2]);
      }
    }
    
    return pairs;
  }
  
  /**
   * Adaptive tournament size based on population diversity
   */
  adaptTournamentSize(state: GEPAState): number {
    const stats = state.getStatistics();
    const diversityRatio = stats.uniquePrograms / stats.totalPrograms;
    
    if (diversityRatio < 0.3) {
      // Low diversity - reduce selection pressure
      return Math.max(2, this.config.tournamentSize - 1);
    } else if (diversityRatio > 0.7) {
      // High diversity - increase selection pressure
      return Math.min(
        state.getProgramCount(),
        this.config.tournamentSize + 1
      );
    }
    
    return this.config.tournamentSize;
  }
}

/**
 * Multi-objective tournament selection for Pareto optimization
 */
export class ParetoTournamentSelector extends TournamentCandidateSelector {
  constructor(config: Omit<TournamentSelectorConfig, 'fitnessFunction'>) {
    super({ ...config, fitnessFunction: 'validation' });
  }
  
  /**
   * Override to use Pareto dominance in tournaments
   */
  selectCandidates(
    state: GEPAState,
    componentsToUpdate: string[],
    numCandidates: number = 1
  ): ComponentMap[] {
    const selected: ComponentMap[] = [];
    
    // Get all Pareto-optimal candidates
    const paretoIndices = new Set<number>();
    for (const instanceParetoSet of state.programAtParetoFrontValset) {
      for (const idx of instanceParetoSet) {
        paretoIndices.add(idx);
      }
    }
    
    if (paretoIndices.size === 0) {
      // Fallback to regular tournament if no Pareto front
      return super.selectCandidates(state, componentsToUpdate, numCandidates);
    }
    
    // Convert to programs
    const paretoPrograms = Array.from(paretoIndices).map(idx => ({
      program: state.getProgram(idx),
      index: idx,
      dominanceCount: this.countDominance(state, idx)
    }));
    
    // Sort by dominance count (how many instances they dominate on)
    paretoPrograms.sort((a, b) => b.dominanceCount - a.dominanceCount);
    
    // Select from Pareto front
    for (let i = 0; i < numCandidates; i++) {
      if (i < paretoPrograms.length) {
        selected.push(paretoPrograms[i].program);
      } else {
        // If need more than Pareto front size, cycle
        selected.push(paretoPrograms[i % paretoPrograms.length].program);
      }
    }
    
    return selected;
  }
  
  /**
   * Count how many validation instances this program dominates on
   */
  private countDominance(state: GEPAState, programIndex: number): number {
    let count = 0;
    
    for (const instanceParetoSet of state.programAtParetoFrontValset) {
      if (instanceParetoSet.has(programIndex)) {
        count++;
      }
    }
    
    return count;
  }
}

/**
 * Factory function for creating tournament selectors
 */
export function createTournamentSelector(
  type: 'standard' | 'pareto' = 'standard',
  config: TournamentSelectorConfig
): CandidateSelector {
  if (type === 'pareto') {
    return new ParetoTournamentSelector(config);
  }
  return new TournamentCandidateSelector(config);
}