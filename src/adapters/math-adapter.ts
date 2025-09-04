/**
 * Math Adapter for GEPA - Optimizes mathematical problem solving
 */

import { GEPAAdapter } from '../core/interfaces.js';
import { GEPACandidate, GEPAResult } from '../core/types.js';

export interface MathExample {
  problem: string;
  answer: string;
}

export interface MathCandidate extends GEPACandidate {
  prompt: string;
}

export interface MathAdapterConfig {
  evaluator: (prompt: string, example: MathExample) => Promise<number>;
  batchSize?: number;
}

export class MathAdapter implements GEPAAdapter<MathExample, MathCandidate> {
  private config: MathAdapterConfig;

  constructor(config: MathAdapterConfig) {
    this.config = {
      batchSize: 1,
      ...config
    };
  }

  async evaluate(
    candidate: MathCandidate,
    examples: MathExample[],
    batchSize: number = this.config.batchSize || 1
  ): Promise<GEPAResult> {
    const scores: number[] = [];
    const predictions: any[] = [];
    const startTime = Date.now();

    for (let i = 0; i < examples.length; i += batchSize) {
      const batch = examples.slice(i, i + batchSize);
      
      for (const example of batch) {
        try {
          const score = await this.config.evaluator(candidate.prompt, example);
          scores.push(score);
          predictions.push({ example, score });
        } catch (error) {
          scores.push(0);
          predictions.push({ example, score: 0, error });
        }
      }
    }

    const accuracy = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const duration = Date.now() - startTime;

    return {
      accuracy,
      scores,
      predictions,
      metadata: {
        duration,
        totalExamples: examples.length,
        successfulPredictions: scores.filter(s => s > 0).length
      }
    };
  }

  async mutate(candidate: MathCandidate, mutationRate: number): Promise<MathCandidate[]> {
    const mutations: MathCandidate[] = [];
    
    if (Math.random() < mutationRate) {
      mutations.push({
        ...candidate,
        prompt: `${candidate.prompt} Step by step:`
      });
    }

    if (Math.random() < mutationRate) {
      mutations.push({
        ...candidate,
        prompt: `Solve this math problem carefully: ${candidate.prompt}`
      });
    }

    return mutations.length > 0 ? mutations : [candidate];
  }

  createInitialPopulation(baseCandidate: MathCandidate, size: number): MathCandidate[] {
    const population: MathCandidate[] = [baseCandidate];
    
    const variations = [
      'Calculate the result',
      'Solve this step by step',
      'Find the answer to',
      'Compute the value',
      'Work out the solution'
    ];

    for (let i = 1; i < size && i - 1 < variations.length; i++) {
      population.push({
        ...baseCandidate,
        prompt: variations[i - 1]
      });
    }

    // Fill remaining slots with variations (removed async/await)
    while (population.length < size) {
      const base = population[Math.floor(Math.random() * population.length)];
      population.push({
        ...base,
        prompt: `${base.prompt} (variant ${population.length})`
      });
    }

    return population.slice(0, size);
  }
}

export default MathAdapter;