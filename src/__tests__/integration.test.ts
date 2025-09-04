import { describe, it, expect } from 'vitest';
import { optimize } from '../api.js';
import { DefaultAdapter } from '../adapters/default-adapter.js';
import { GEPAAdapter, EvaluationBatch } from '../core/adapter.js';
import { ComponentMap, ReflectiveDataset } from '../types/index.js';
import { SilentLogger } from '../logging/logger.js';

describe('GEPA Integration Tests', () => {
  // Mock adapter for controlled testing
  class TestAdapter implements GEPAAdapter {
    private iteration = 0;
    
    evaluate(
      batch: any[],
      candidate: ComponentMap,
      captureTraces: boolean = false
    ): EvaluationBatch {
      const outputs: any[] = [];
      const scores: number[] = [];
      const trajectories = captureTraces ? [] : null;
      
      // Simulate improvement based on prompt quality
      const promptQuality = candidate.system_prompt?.length || 0;
      const baseScore = Math.min(0.3 + (promptQuality / 100), 0.9);
      
      for (const item of batch) {
        const score = baseScore + (Math.random() * 0.1);
        outputs.push({ result: `Answer: ${score}` });
        scores.push(score);
        
        if (trajectories) {
          trajectories.push({
            input: item,
            output: `Answer: ${score}`,
            score
          });
        }
      }
      
      return { outputs, scores, trajectories };
    }
    
    makeReflectiveDataset(
      candidate: ComponentMap,
      evalBatch: EvaluationBatch,
      componentsToUpdate: string[]
    ): ReflectiveDataset {
      const dataset: ReflectiveDataset = {};
      
      for (const component of componentsToUpdate) {
        dataset[component] = evalBatch.trajectories?.map((traj, idx) => ({
          input: traj.input,
          output: traj.output,
          score: evalBatch.scores[idx],
          feedback: evalBatch.scores[idx] < 0.5 ? 'Poor performance' : 'Good'
        })) || [];
      }
      
      return dataset;
    }
    
    proposeNewTexts(
      candidate: ComponentMap,
      reflectiveDataset: ReflectiveDataset,
      componentsToUpdate: string[]
    ): ComponentMap {
      this.iteration++;
      const newTexts: ComponentMap = {};
      
      for (const component of componentsToUpdate) {
        // Simulate improvement by making prompt longer/better
        const current = candidate[component] || '';
        newTexts[component] = current + ` [Improved v${this.iteration}]`;
      }
      
      return newTexts;
    }
  }
  
  it('should optimize a candidate over multiple iterations', async () => {
    const trainset = [
      { question: 'Q1', answer: 'A1' },
      { question: 'Q2', answer: 'A2' },
      { question: 'Q3', answer: 'A3' }
    ];
    
    const seedCandidate = {
      system_prompt: 'Initial prompt'
    };
    
    const adapter = new TestAdapter();
    
    const result = await optimize({
      seedCandidate,
      trainset,
      adapter,
      maxMetricCalls: 30,
      reflectionMinibatchSize: 2,
      candidateSelectionStrategy: 'current-best',
      logger: new SilentLogger(),
      runDir: null,
      seed: 42
    });
    
    // Verify optimization occurred
    expect(result.bestScore).toBeGreaterThan(0);
    expect(result.allCandidates.length).toBeGreaterThan(1);
    expect(result.totalEvaluations).toBeLessThanOrEqual(35); // Allow some buffer for initial eval
    
    // Verify improvement
    const initialScore = result.allScores[0];
    expect(result.bestScore).toBeGreaterThanOrEqual(initialScore);
    
    // Verify best candidate has been modified
    expect(result.bestCandidate.system_prompt).not.toBe(seedCandidate.system_prompt);
    expect(result.bestCandidate.system_prompt).toContain('Improved');
  });
  
  it('should support pareto-based candidate selection', async () => {
    const trainset = Array(10).fill(0).map((_, i) => ({
      id: i,
      data: `item${i}`
    }));
    
    const seedCandidate = {
      component1: 'Initial text'
    };
    
    const adapter = new TestAdapter();
    
    const result = await optimize({
      seedCandidate,
      trainset,
      adapter,
      maxMetricCalls: 50,
      candidateSelectionStrategy: 'pareto',
      reflectionMinibatchSize: 3,
      logger: new SilentLogger(),
      runDir: null
    });
    
    // Verify pareto front candidates exist
    expect(result.paretoFrontCandidates.length).toBeGreaterThan(0);
    expect(result.paretoFrontCandidates).toContainEqual(result.bestCandidate);
  });
  
  it('should handle merge proposer when enabled', async () => {
    const trainset = Array(5).fill(0).map((_, i) => ({
      id: i,
      value: i * 10
    }));
    
    const valset = Array(3).fill(0).map((_, i) => ({
      id: i + 5,
      value: (i + 5) * 10
    }));
    
    const seedCandidate = {
      prompt1: 'Initial prompt 1',
      prompt2: 'Initial prompt 2'
    };
    
    const adapter = new TestAdapter();
    
    const result = await optimize({
      seedCandidate,
      trainset,
      valset,
      adapter,
      maxMetricCalls: 100,
      useMerge: true,
      maxMergeInvocations: 3,
      candidateSelectionStrategy: 'pareto',
      reflectionMinibatchSize: 2,
      logger: new SilentLogger(),
      runDir: null,
      seed: 123
    });
    
    // Verify optimization with merge
    expect(result.allCandidates.length).toBeGreaterThan(1);
    expect(result.bestScore).toBeGreaterThan(0);
    
    // With merge enabled, we should see exploration of pareto front
    expect(result.paretoFrontCandidates.length).toBeGreaterThan(0);
  });
  
  it('should respect perfect score early stopping', async () => {
    class PerfectScoreAdapter implements GEPAAdapter {
      evaluate(batch: any[], candidate: ComponentMap, captureTraces: boolean): EvaluationBatch {
        // Always return perfect scores
        const outputs = batch.map(() => ({ result: 'perfect' }));
        const scores = batch.map(() => 1.0);
        const trajectories = captureTraces ? 
          batch.map(b => ({ input: b, output: 'perfect' })) : null;
        
        return { outputs, scores, trajectories };
      }
      
      makeReflectiveDataset(): ReflectiveDataset {
        return { component: [] };
      }
      
      proposeNewTexts(candidate: ComponentMap): ComponentMap {
        return { ...candidate, updated: 'should not happen' };
      }
    }
    
    const trainset = [{ data: 1 }, { data: 2 }];
    const seedCandidate = { prompt: 'test' };
    
    const result = await optimize({
      seedCandidate,
      trainset,
      adapter: new PerfectScoreAdapter(),
      maxMetricCalls: 50,
      perfectScore: 1.0,
      skipPerfectScore: true,
      logger: new SilentLogger(),
      runDir: null
    });
    
    // Should stop early due to perfect scores
    expect(result.bestScore).toBe(1.0);
    expect(result.allCandidates.length).toBe(1); // Only seed candidate
  });
  
  it('should track outputs when requested', async () => {
    const trainset = [{ id: 1 }, { id: 2 }];
    const valset = [{ id: 3 }, { id: 4 }];
    
    const adapter = new TestAdapter();
    
    const result = await optimize({
      seedCandidate: { prompt: 'test' },
      trainset,
      valset,
      adapter,
      maxMetricCalls: 20,
      trackBestOutputs: true,
      logger: new SilentLogger(),
      runDir: null
    });
    
    // Should have best outputs when tracking is enabled
    expect(result.bestOutputs).toBeDefined();
    if (result.bestOutputs) {
      expect(result.bestOutputs.length).toBe(valset.length);
    }
  });
});