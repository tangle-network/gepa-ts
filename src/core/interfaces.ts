/**
 * Core interfaces for GEPA optimization system
 */

import { GEPACandidate, GEPAResult } from './types.js';

export interface GEPAAdapter<TExample, TCandidate extends GEPACandidate = GEPACandidate> {
  /**
   * Evaluate a candidate on a batch of examples
   */
  evaluate(
    candidate: TCandidate,
    examples: TExample[],
    batchSize?: number
  ): Promise<GEPAResult>;

  /**
   * Mutate a candidate to create variations
   */
  mutate(candidate: TCandidate, mutationRate: number): Promise<TCandidate[]>;

  /**
   * Crossover two candidates to create offspring
   */
  crossover?(parent1: TCandidate, parent2: TCandidate): Promise<TCandidate[]>;

  /**
   * Create initial population from base candidate
   */
  createInitialPopulation(baseCandidate: TCandidate, size: number): TCandidate[];
}

export interface LanguageModel {
  /**
   * Generate text completion
   */
  generate(prompt: string): Promise<string>;

  /**
   * Get model configuration
   */
  getConfig(): Record<string, any>;
}

export interface OptimizerConfig {
  populationSize: number;
  generations: number;
  mutationRate: number;
  crossoverRate?: number;
  verbose?: boolean;
}

export interface OptimizationRequest<TExample, TCandidate extends GEPACandidate = GEPACandidate> {
  initialCandidate: TCandidate;
  trainData: TExample[];
  valData: TExample[];
  componentsToUpdate: string[];
}

export interface OptimizationResult<TCandidate extends GEPACandidate = GEPACandidate> {
  bestCandidate: TCandidate;
  bestScore: number;
  generationsCompleted: number;
  totalEvaluations: number;
  history: Array<{
    generation: number;
    bestScore: number;
    averageScore: number;
    candidates: TCandidate[];
  }>;
}

export default GEPAAdapter;