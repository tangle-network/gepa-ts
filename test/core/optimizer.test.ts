/**
 * Comprehensive tests for GEPAOptimizer with real optimization scenarios
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';
import { GEPAOptimizer } from '../../src/core/optimizer.js';
import { DefaultAdapter } from '../../src/adapters/default-adapter.js';
import { TestLLM, RecordReplayManager } from '../infrastructure/record-replay.js';
import { ComponentMap } from '../../src/types/index.js';
import { tmpdir } from 'os';
import { join } from 'path';
import { rmSync, existsSync } from 'fs';

interface TestExample {
  input: string;
  expectedOutput: string;
  score?: number;
}

describe('GEPAOptimizer', () => {
  let testLLM: TestLLM;
  let tempDir: string;
  
  beforeEach(() => {
    testLLM = new TestLLM({ mode: 'replay' });
    tempDir = join(tmpdir(), `gepa-test-${Date.now()}`);
  });
  
  afterEach(() => {
    testLLM.finish();
    if (existsSync(tempDir)) {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
  
  describe('Basic Optimization', () => {
    it('should optimize a simple prompt', async () => {
      // Create evaluator that rewards specific keywords
      const evaluator = async (prompt: string, example: TestExample): Promise<number> => {
        const response = await testLLM.generate(`${prompt}\n\nInput: ${example.input}`);
        
        // Score based on presence of expected patterns
        let score = 0;
        if (response.toLowerCase().includes(example.expectedOutput.toLowerCase())) {
          score += 0.5;
        }
        if (prompt.includes('clear') || prompt.includes('specific')) {
          score += 0.3;
        }
        if (prompt.length > 20 && prompt.length < 200) {
          score += 0.2;
        }
        
        return Math.min(1.0, score);
      };
      
      const adapter = new DefaultAdapter<TestExample>({
        evaluator,
        batchSize: 2
      });
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 4,
          generations: 2,
          mutationRate: 0.5,
          mergeRate: 0.3,
          verbose: false,
          runDir: tempDir
        }
      });
      
      const trainData: TestExample[] = [
        { input: 'hello', expectedOutput: 'greeting' },
        { input: 'goodbye', expectedOutput: 'farewell' }
      ];
      
      const valData: TestExample[] = [
        { input: 'hi', expectedOutput: 'greeting' },
        { input: 'bye', expectedOutput: 'farewell' }
      ];
      
      const initialCandidate = {
        instruction: 'Classify the input text.',
        format: 'Output a single word.'
      };
      
      const result = await optimizer.optimize({
        initialCandidate,
        trainData,
        valData,
        componentsToUpdate: ['instruction']
      });
      
      expect(result).toBeDefined();
      expect(result.bestScore).toBeGreaterThanOrEqual(0);
      expect(result.bestScore).toBeLessThanOrEqual(1);
      expect(result.bestCandidate).toHaveProperty('instruction');
      expect(result.bestCandidate).toHaveProperty('format');
      expect(result.generationsCompleted).toBeGreaterThan(0);
      expect(result.totalEvaluations).toBeGreaterThan(0);
      expect(result.history.length).toBeGreaterThan(0);
    });
    
    it('should handle multiple components', async () => {
      const evaluator = async (prompt: string, example: TestExample): Promise<number> => {
        // Combine all component texts
        const fullPrompt = Object.values(prompt).join('\n');
        const response = await testLLM.generate(`${fullPrompt}\n\nInput: ${example.input}`);
        
        return example.score || 0.5;
      };
      
      const adapter = new DefaultAdapter<TestExample>({
        evaluator: async (components: any, example: TestExample) => {
          const prompt = typeof components === 'string' 
            ? components 
            : Object.values(components).join('\n');
          return evaluator(prompt, example);
        },
        batchSize: 1
      });
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 3,
          generations: 1,
          mutationRate: 0.4,
          mergeRate: 0.2,
          verbose: false
        }
      });
      
      const data: TestExample[] = [
        { input: 'test1', expectedOutput: 'result1', score: 0.6 },
        { input: 'test2', expectedOutput: 'result2', score: 0.7 }
      ];
      
      const result = await optimizer.optimize({
        initialCandidate: {
          component1: 'Initial text 1',
          component2: 'Initial text 2',
          component3: 'Initial text 3'
        },
        trainData: data,
        valData: data,
        componentsToUpdate: ['component1', 'component2']
      });
      
      expect(result.bestCandidate).toHaveProperty('component1');
      expect(result.bestCandidate).toHaveProperty('component2');
      expect(result.bestCandidate).toHaveProperty('component3');
      expect(result.bestCandidate.component3).toBe('Initial text 3'); // Unchanged
    });
  });
  
  describe('Pareto Front Tracking', () => {
    it('should track per-instance Pareto fronts correctly', async () => {
      // This tests the critical fix for per-instance Pareto tracking
      const instanceScores = new Map<string, number[]>();
      
      const evaluator = async (prompt: string, example: TestExample): Promise<number> => {
        // Each instance gets different scores for different prompts
        const promptHash = prompt.length % 3; // Simple hash
        const instanceId = example.input;
        
        if (!instanceScores.has(instanceId)) {
          instanceScores.set(instanceId, [0.3, 0.5, 0.7]);
        }
        
        const scores = instanceScores.get(instanceId)!;
        return scores[promptHash];
      };
      
      const adapter = new DefaultAdapter<TestExample>({
        evaluator: async (components: any, example: TestExample) => {
          const prompt = typeof components === 'string' 
            ? components 
            : Object.values(components).join(' ');
          return evaluator(prompt, example);
        },
        batchSize: 1
      });
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 6,
          generations: 3,
          mutationRate: 0.3,
          mergeRate: 0.3,
          tournamentSize: 2,
          verbose: false
        }
      });
      
      const trainData: TestExample[] = [
        { input: 'instance1', expectedOutput: 'out1' },
        { input: 'instance2', expectedOutput: 'out2' }
      ];
      
      const valData: TestExample[] = [
        { input: 'val1', expectedOutput: 'vout1' },
        { input: 'val2', expectedOutput: 'vout2' },
        { input: 'val3', expectedOutput: 'vout3' }
      ];
      
      const result = await optimizer.optimize({
        initialCandidate: {
          prompt: 'Initial prompt text'
        },
        trainData,
        valData,
        componentsToUpdate: ['prompt']
      });
      
      // Check that we have history
      expect(result.history.length).toBeGreaterThan(0);
      
      // Verify Pareto front was tracked
      const finalGen = result.history[result.history.length - 1];
      if (finalGen.paretoFront) {
        // Each validation instance should have its own Pareto score
        expect(finalGen.paretoFront.length).toBe(valData.length);
      }
    });
  });
  
  describe('Mutation and Merging', () => {
    it('should apply mutations to create diversity', async () => {
      const seenPrompts = new Set<string>();
      
      const evaluator = async (prompt: string, example: TestExample): Promise<number> => {
        seenPrompts.add(prompt);
        return Math.random() * 0.5 + 0.25; // Random score between 0.25 and 0.75
      };
      
      const adapter = new DefaultAdapter<TestExample>({
        evaluator: async (components: any, example: TestExample) => {
          const prompt = typeof components === 'string' 
            ? components 
            : components.text;
          return evaluator(prompt, example);
        },
        batchSize: 1
      });
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 8,
          generations: 2,
          mutationRate: 0.7, // High mutation rate
          mergeRate: 0.1,    // Low merge rate
          verbose: false
        }
      });
      
      const data: TestExample[] = [
        { input: 'test', expectedOutput: 'result' }
      ];
      
      const result = await optimizer.optimize({
        initialCandidate: {
          text: 'This is the initial prompt text that should be mutated.'
        },
        trainData: data,
        valData: data,
        componentsToUpdate: ['text']
      });
      
      // Should have created multiple different prompts through mutation
      expect(seenPrompts.size).toBeGreaterThan(3);
      
      // Final prompt should be different from initial
      expect(result.bestCandidate.text).not.toBe('This is the initial prompt text that should be mutated.');
    });
    
    it('should merge candidates to combine good features', async () => {
      let mergeCount = 0;
      
      const evaluator = async (prompt: string, example: TestExample): Promise<number> => {
        // Reward prompts that have both "precise" and "clear"
        let score = 0.3;
        if (prompt.includes('precise')) score += 0.35;
        if (prompt.includes('clear')) score += 0.35;
        return score;
      };
      
      const adapter = new DefaultAdapter<TestExample>({
        evaluator: async (components: any, example: TestExample) => {
          const prompt = components.instruction || '';
          
          // Detect merges (prompts with both keywords)
          if (prompt.includes('precise') && prompt.includes('clear')) {
            mergeCount++;
          }
          
          return evaluator(prompt, example);
        },
        batchSize: 1
      });
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 6,
          generations: 3,
          mutationRate: 0.2,
          mergeRate: 0.6, // High merge rate
          verbose: false
        }
      });
      
      const data: TestExample[] = [
        { input: 'test', expectedOutput: 'result' }
      ];
      
      // Seed with prompts containing different good features
      const result = await optimizer.optimize({
        initialCandidate: {
          instruction: 'Be precise in your response.'
        },
        trainData: data,
        valData: data,
        componentsToUpdate: ['instruction'],
        seedPopulation: [
          { instruction: 'Be precise in your response.' },
          { instruction: 'Provide clear explanations.' }
        ]
      });
      
      // Should have performed merges
      expect(mergeCount).toBeGreaterThan(0);
      
      // Best candidate should ideally have high score
      expect(result.bestScore).toBeGreaterThan(0.5);
    });
  });
  
  describe('Error Handling', () => {
    it('should handle evaluation failures gracefully', async () => {
      let callCount = 0;
      
      const evaluator = async (prompt: string, example: TestExample): Promise<number> => {
        callCount++;
        // Fail on some evaluations
        if (callCount % 3 === 0) {
          throw new Error('Simulated evaluation failure');
        }
        return 0.5;
      };
      
      const adapter = new DefaultAdapter<TestExample>({
        evaluator: async (components: any, example: TestExample) => {
          return evaluator(components.text || '', example);
        },
        batchSize: 1
      });
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 3,
          generations: 1,
          mutationRate: 0.3,
          verbose: false
        }
      });
      
      const data: TestExample[] = [
        { input: 'test1', expectedOutput: 'result1' },
        { input: 'test2', expectedOutput: 'result2' }
      ];
      
      // Should complete despite some failures
      const result = await optimizer.optimize({
        initialCandidate: {
          text: 'Initial text'
        },
        trainData: data,
        valData: data,
        componentsToUpdate: ['text']
      });
      
      expect(result).toBeDefined();
      expect(result.generationsCompleted).toBeGreaterThan(0);
    });
    
    it('should handle empty datasets', async () => {
      const adapter = new DefaultAdapter<TestExample>({
        evaluator: async () => 0.5,
        batchSize: 1
      });
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 2,
          generations: 1,
          verbose: false
        }
      });
      
      const result = await optimizer.optimize({
        initialCandidate: { text: 'Test' },
        trainData: [],
        valData: [],
        componentsToUpdate: ['text']
      });
      
      expect(result).toBeDefined();
      expect(result.bestScore).toBe(0); // No validation data
    });
  });
  
  describe('Configuration Options', () => {
    it('should respect population size configuration', async () => {
      let evaluationCount = 0;
      
      const adapter = new DefaultAdapter<TestExample>({
        evaluator: async () => {
          evaluationCount++;
          return 0.5;
        },
        batchSize: 1
      });
      
      const populationSize = 5;
      const generations = 2;
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize,
          generations,
          mutationRate: 0.3,
          verbose: false
        }
      });
      
      const data: TestExample[] = [
        { input: 'test', expectedOutput: 'result' }
      ];
      
      await optimizer.optimize({
        initialCandidate: { text: 'Test' },
        trainData: data,
        valData: data,
        componentsToUpdate: ['text']
      });
      
      // Should evaluate approximately populationSize * generations * datasets
      const expectedMin = populationSize * generations;
      const expectedMax = populationSize * generations * 2 * 2; // With train and val
      
      expect(evaluationCount).toBeGreaterThanOrEqual(expectedMin);
      expect(evaluationCount).toBeLessThanOrEqual(expectedMax);
    });
    
    it('should apply elite ratio correctly', async () => {
      const scores: number[] = [];
      
      const adapter = new DefaultAdapter<TestExample>({
        evaluator: async (components: any) => {
          // Assign decreasing scores to track elites
          const score = 1.0 - (scores.length * 0.1);
          scores.push(score);
          return Math.max(0, score);
        },
        batchSize: 1
      });
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 10,
          generations: 2,
          eliteRatio: 0.2, // Keep top 20%
          mutationRate: 0.1,
          verbose: false
        }
      });
      
      const data: TestExample[] = [
        { input: 'test', expectedOutput: 'result' }
      ];
      
      const result = await optimizer.optimize({
        initialCandidate: { text: 'Test' },
        trainData: data,
        valData: data,
        componentsToUpdate: ['text']
      });
      
      // Best score should be from early evaluations (elites preserved)
      expect(result.bestScore).toBeGreaterThan(0.7);
    });
  });
  
  describe('Integration with Adapters', () => {
    it('should work with custom adapter implementations', async () => {
      // Custom adapter that tracks component updates
      class TrackingAdapter extends DefaultAdapter<TestExample> {
        public updatedComponents: Set<string> = new Set();
        
        constructor() {
          super({
            evaluator: async (components: any, example: TestExample) => {
              // Track which components were updated
              for (const key of Object.keys(components)) {
                if (components[key] !== 'Initial') {
                  this.updatedComponents.add(key);
                }
              }
              return 0.5;
            },
            batchSize: 1
          });
        }
      }
      
      const adapter = new TrackingAdapter();
      
      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 4,
          generations: 2,
          mutationRate: 0.5,
          verbose: false
        }
      });
      
      const data: TestExample[] = [
        { input: 'test', expectedOutput: 'result' }
      ];
      
      await optimizer.optimize({
        initialCandidate: {
          comp1: 'Initial',
          comp2: 'Initial',
          comp3: 'Initial'
        },
        trainData: data,
        valData: data,
        componentsToUpdate: ['comp1', 'comp2']
      });
      
      // Should only have updated specified components
      expect(adapter.updatedComponents.has('comp1')).toBe(true);
      expect(adapter.updatedComponents.has('comp2')).toBe(true);
      expect(adapter.updatedComponents.has('comp3')).toBe(false);
    });
  });
});