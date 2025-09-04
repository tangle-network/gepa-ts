/**
 * GEPA Optimizer - Main optimization engine
 */

import { 
  GEPAAdapter, 
  OptimizerConfig, 
  OptimizationRequest, 
  OptimizationResult 
} from './interfaces.js';
import { GEPACandidate } from './types.js';

export class GEPAOptimizer<TExample = any, TCandidate extends GEPACandidate = GEPACandidate> {
  private adapter: GEPAAdapter<TExample, TCandidate>;
  private config: OptimizerConfig;

  constructor(options: {
    adapter: GEPAAdapter<TExample, TCandidate>;
    config: OptimizerConfig;
  }) {
    this.adapter = options.adapter;
    this.config = options.config;
  }

  async optimize(
    request: OptimizationRequest<TExample, TCandidate>
  ): Promise<OptimizationResult<TCandidate>> {
    if (this.config.verbose) {
      console.log('🚀 Starting GEPA optimization...');
      console.log(`   Population: ${this.config.populationSize}`);
      console.log(`   Generations: ${this.config.generations}`);
      console.log(`   Mutation rate: ${this.config.mutationRate}`);
    }

    // Create initial population
    const population = this.adapter.createInitialPopulation(
      request.initialCandidate,
      this.config.populationSize
    );

    let bestCandidate = request.initialCandidate;
    let bestScore = 0;
    const history: Array<{
      generation: number;
      bestScore: number;
      averageScore: number;
      candidates: TCandidate[];
    }> = [];

    let totalEvaluations = 0;

    // Evolution loop
    for (let generation = 0; generation < this.config.generations; generation++) {
      if (this.config.verbose) {
        console.log(`\n🧬 Generation ${generation + 1}/${this.config.generations}`);
      }

      const scores: number[] = [];
      
      // Evaluate population
      for (const candidate of population) {
        const result = await this.adapter.evaluate(candidate, request.valData);
        scores.push(result.accuracy);
        totalEvaluations++;

        // Update best
        if (result.accuracy > bestScore) {
          bestScore = result.accuracy;
          bestCandidate = candidate;
          
          if (this.config.verbose) {
            console.log(`   🎯 New best: ${(bestScore * 100).toFixed(1)}%`);
          }
        }
      }

      const averageScore = scores.reduce((sum, s) => sum + s, 0) / scores.length;
      
      if (this.config.verbose) {
        console.log(`   📊 Average: ${(averageScore * 100).toFixed(1)}%`);
      }

      history.push({
        generation: generation + 1,
        bestScore,
        averageScore,
        candidates: [...population]
      });

      // Create next generation (simplified)
      if (generation < this.config.generations - 1) {
        const nextGeneration: TCandidate[] = [];
        
        // Keep best candidate (elitism)
        nextGeneration.push(bestCandidate);
        
        // Generate mutations
        for (let i = 1; i < this.config.populationSize; i++) {
          const parent = population[Math.floor(Math.random() * population.length)];
          const mutations = await this.adapter.mutate(parent, this.config.mutationRate);
          nextGeneration.push(mutations[0] || parent);
        }
        
        // Replace population
        population.splice(0, population.length, ...nextGeneration);
      }
    }

    if (this.config.verbose) {
      console.log(`\n✅ Optimization complete!`);
      console.log(`🏆 Best score: ${(bestScore * 100).toFixed(1)}%`);
      console.log(`📈 Total evaluations: ${totalEvaluations}`);
    }

    return {
      bestCandidate,
      bestScore,
      generationsCompleted: this.config.generations,
      totalEvaluations,
      history
    };
  }
}

export default GEPAOptimizer;