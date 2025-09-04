/**
 * Terminal Adapter for GEPA - Optimizes terminal command generation
 */

import { GEPAAdapter } from '../core/interfaces.js';
import { GEPACandidate, GEPAResult } from '../core/types.js';

export interface TerminalExample {
  command: string;
  answer: string;
}

export interface TerminalCandidate extends GEPACandidate {
  prompt: string;
}

export interface TerminalAdapterConfig {
  evaluator: (prompt: string, example: TerminalExample) => Promise<number>;
  batchSize?: number;
}

export class TerminalAdapter implements GEPAAdapter<TerminalExample, TerminalCandidate> {
  private config: TerminalAdapterConfig;

  constructor(config: TerminalAdapterConfig) {
    this.config = {
      batchSize: 1,
      ...config
    };
  }

  async evaluate(
    candidate: TerminalCandidate,
    examples: TerminalExample[],
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

  async mutate(candidate: TerminalCandidate, mutationRate: number): Promise<TerminalCandidate[]> {
    const mutations: TerminalCandidate[] = [];
    
    if (Math.random() < mutationRate) {
      mutations.push({
        ...candidate,
        prompt: `${candidate.prompt} (provide exact command)`
      });
    }

    if (Math.random() < mutationRate) {
      mutations.push({
        ...candidate,
        prompt: `Terminal command for: ${candidate.prompt}`
      });
    }

    if (Math.random() < mutationRate) {
      mutations.push({
        ...candidate,
        prompt: `Unix/Linux command to ${candidate.prompt.toLowerCase()}`
      });
    }

    return mutations.length > 0 ? mutations : [candidate];
  }

  createInitialPopulation(baseCandidate: TerminalCandidate, size: number): TerminalCandidate[] {
    const population: TerminalCandidate[] = [baseCandidate];
    
    const variations = [
      'What command should I use',
      'Terminal command for',
      'Unix command to',
      'Shell command for',
      'Bash command to'
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

export default TerminalAdapter;