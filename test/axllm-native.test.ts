import { describe, it, expect, beforeEach, vi } from 'vitest';
import { AxLLMNativeAdapter, createAxLLMNativeAdapter } from '../src/adapters/axllm-native-adapter.js';
import { GEPAOptimizer } from '../src/core/optimizer.js';

describe('AxLLM Native Adapter Tests', () => {
  let mockAx: any;
  let mockLlm: any;
  let mockAxProgram: any;

  beforeEach(() => {
    // Mock AxLLM ax() function
    mockAxProgram = {
      forward: vi.fn(),
      setDemos: vi.fn(),
      applyOptimization: vi.fn(),
      _signature: 'input:string -> output:string'
    };
    
    mockAx = vi.fn().mockReturnValue(mockAxProgram);
    
    // Mock AI instance
    mockLlm = {
      generate: vi.fn().mockResolvedValue('Generated response'),
      config: { model: 'gpt-4o-mini' }
    };
  });

  describe('Basic Functionality', () => {
    it('should create adapter with default program', () => {
      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: ({ prediction, example }) => {
          return prediction.output === example.output ? 1 : 0;
        }
      });

      expect(adapter).toBeInstanceOf(AxLLMNativeAdapter);
    });

    it('should parse AxLLM signature strings', () => {
      const adapter = new AxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        baseProgram: {
          signature: 'reviewText:string "Customer review" -> sentiment:class "positive,negative,neutral" "How they feel"'
        },
        metricFn: ({ prediction, example }) => {
          return prediction.sentiment === example.sentiment ? 1 : 0;
        }
      });

      // Test internal signature parsing
      const parsed = (adapter as any).parseSignature(
        'reviewText:string "Customer review" -> sentiment:class "positive,negative,neutral" "How they feel"'
      );

      expect(parsed.inputs).toHaveProperty('reviewText');
      expect(parsed.outputs).toHaveProperty('sentiment');
      expect(parsed.outputs.sentiment.classes).toEqual(['positive', 'negative', 'neutral']);
    });

    it('should evaluate batch of examples', async () => {
      mockAxProgram.forward.mockResolvedValue({ sentiment: 'positive' });

      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: ({ prediction, example }) => {
          return prediction.sentiment === example.sentiment ? 1 : 0;
        }
      });

      const batch = [
        { reviewText: 'Great product!', sentiment: 'positive' },
        { reviewText: 'Terrible!', sentiment: 'negative' }
      ];

      const result = await adapter.evaluate(
        batch,
        { program: JSON.stringify({ signature: 'reviewText:string -> sentiment:string' }) },
        false
      );

      expect(result.outputs).toHaveLength(2);
      expect(result.scores).toHaveLength(2);
      expect(result.scores[0]).toBe(1); // First matches
      expect(result.scores[1]).toBe(0); // Second doesn't match
    });

    it('should capture traces when requested', async () => {
      mockAxProgram.forward.mockResolvedValue({ output: 'result' });

      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: () => 1
      });

      const result = await adapter.evaluate(
        [{ input: 'test' }],
        { program: JSON.stringify({ signature: 'input:string -> output:string' }) },
        true // Capture traces
      );

      expect(result.trajectories).not.toBeNull();
      expect(result.trajectories).toHaveLength(1);
      expect(result.trajectories![0]).toHaveProperty('signature');
      expect(result.trajectories![0]).toHaveProperty('inputs');
      expect(result.trajectories![0]).toHaveProperty('outputs');
      expect(result.trajectories![0]).toHaveProperty('score');
    });
  });

  describe('Program Evolution', () => {
    it('should make reflective dataset from evaluation', async () => {
      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: () => 0.5
      });

      const evalBatch = {
        outputs: [{ sentiment: 'positive', _score: 0.5 }],
        scores: [0.5],
        trajectories: [{
          signature: 'review -> sentiment',
          inputs: { review: 'Good' },
          outputs: { sentiment: 'positive' },
          modelCalls: [],
          score: 0.5
        }]
      };

      const reflectiveDataset = await adapter.makeReflectiveDataset(
        { program: '{}' },
        evalBatch,
        ['program']
      );

      expect(reflectiveDataset).toHaveProperty('program');
      expect(reflectiveDataset['program']).toBeInstanceOf(Array);
      expect(reflectiveDataset['program'][0]).toHaveProperty('Score', 0.5);
      expect(reflectiveDataset['program'][0]).toHaveProperty('Feedback');
      expect(reflectiveDataset['program'][0]).toHaveProperty('Suggestions');
    });

    it('should propose evolved programs based on performance', async () => {
      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: () => 0.3,
        reflectionLm: async (prompt) => {
          // Mock LLM evolution
          return JSON.stringify({
            signature: 'reviewText:string "Analyze carefully" -> sentiment:class "positive,negative,neutral" "Classification"',
            demos: [{ reviewText: 'Great!', sentiment: 'positive' }],
            modelConfig: { temperature: 0.3 }
          });
        }
      });

      const reflectiveDataset = {
        'program': [
          { Score: 0.3, Feedback: 'Poor', Suggestions: ['Add demos'] }
        ]
      };

      const evolved = await adapter.proposeNewTexts(
        { program: JSON.stringify({ signature: 'review -> sentiment' }) },
        reflectiveDataset,
        ['program']
      );

      expect(evolved).toHaveProperty('program');
      const evolvedProgram = JSON.parse(evolved['program']);
      expect(evolvedProgram).toHaveProperty('signature');
      expect(evolvedProgram).toHaveProperty('demos');
    });

    it('should apply different evolution strategies based on score', async () => {
      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: () => 1
      });

      // Test major evolution (score < 0.3)
      const lowScoreDataset = {
        'program': [{ Score: 0.2, Feedback: 'Poor' }]
      };

      const majorEvolved = await adapter.proposeNewTexts(
        { program: JSON.stringify({ signature: 'input -> output' }) },
        lowScoreDataset,
        ['program']
      );

      const majorProgram = JSON.parse(majorEvolved['program']);
      expect(majorProgram.modelConfig?.temperature).toBeLessThan(0.7);

      // Test minor optimization (score > 0.7)
      const highScoreDataset = {
        'program': [{ Score: 0.8, Feedback: 'Good' }]
      };

      const minorEvolved = await adapter.proposeNewTexts(
        { program: JSON.stringify({ signature: 'input -> output', demos: Array(10).fill({}) }) },
        highScoreDataset,
        ['program']
      );

      const minorProgram = JSON.parse(minorEvolved['program']);
      expect(minorProgram.demos?.length).toBeLessThanOrEqual(5);
    });
  });

  describe('AxOptimizedProgram Integration', () => {
    it('should convert to AxOptimizedProgram format', () => {
      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: () => 1
      });

      const candidate = {
        program: JSON.stringify({
          signature: 'input -> output',
          demos: [{ input: 'test', output: 'result' }],
          modelConfig: { temperature: 0.5 }
        })
      };

      const axOptimized = adapter.toAxOptimizedProgram(candidate, 0.95, { rounds: 10 });

      expect(axOptimized).toHaveProperty('bestScore', 0.95);
      expect(axOptimized).toHaveProperty('instruction');
      expect(axOptimized).toHaveProperty('demos');
      expect(axOptimized).toHaveProperty('modelConfig');
      expect(axOptimized).toHaveProperty('optimizerType', 'GEPA');
      expect(axOptimized).toHaveProperty('totalRounds', 10);
      expect(axOptimized).toHaveProperty('converged', false); // 0.95 < 0.95 threshold
    });

    it('should load from AxOptimizedProgram format', () => {
      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: () => 1
      });

      const axOptimized = {
        bestScore: 0.9,
        instruction: 'review:string -> sentiment:string',
        demos: [{ review: 'Good', sentiment: 'positive' }],
        modelConfig: { temperature: 0.3 },
        optimizerType: 'MiPRO',
        optimizationTime: Date.now(),
        totalRounds: 5,
        converged: false
      };

      const candidate = adapter.fromAxOptimizedProgram(axOptimized);

      expect(candidate).toHaveProperty('program');
      const program = JSON.parse(candidate['program']);
      expect(program).toHaveProperty('signature', 'review:string -> sentiment:string');
      expect(program).toHaveProperty('demos');
      expect(program.demos).toHaveLength(1);
    });
  });

  describe('GEPA Integration', () => {
    it('should work with GEPAOptimizer', async () => {
      mockAxProgram.forward.mockResolvedValue({ sentiment: 'positive' });

      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        baseProgram: {
          signature: 'review:string -> sentiment:string'
        },
        metricFn: ({ prediction, example }) => {
          return prediction.sentiment === example.sentiment ? 1 : 0;
        }
      });

      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 2,
          generations: 1,
          mutationRate: 0.5
        }
      });

      const trainData = [
        { review: 'Great!', sentiment: 'positive' },
        { review: 'Bad!', sentiment: 'negative' }
      ];

      const result = await optimizer.optimize({
        initialCandidate: { 
          program: JSON.stringify({ 
            signature: 'review:string -> sentiment:string' 
          }) 
        },
        trainData,
        valData: trainData,
        componentsToUpdate: ['program'],
        maxMetricCalls: 10
      });

      expect(result).toHaveProperty('bestCandidate');
      expect(result.bestCandidate).toHaveProperty('program');
      expect(result).toHaveProperty('bestScore');
      expect(result).toHaveProperty('generation');
    });

    it('should evolve programs across generations', async () => {
      let callCount = 0;
      mockAxProgram.forward.mockImplementation(() => {
        // Improve over time
        callCount++;
        return Promise.resolve({ 
          sentiment: callCount > 5 ? 'positive' : 'negative' 
        });
      });

      const adapter = createAxLLMNativeAdapter({
        axInstance: mockAx,
        llm: mockLlm,
        metricFn: ({ prediction, example }) => {
          return prediction.sentiment === example.sentiment ? 1 : 0;
        },
        reflectionLm: async (prompt) => {
          // Evolve the program
          return JSON.stringify({
            signature: 'review:string "Enhanced" -> sentiment:string "Better"',
            demos: [{ review: 'Example', sentiment: 'positive' }]
          });
        }
      });

      const optimizer = new GEPAOptimizer({
        adapter,
        config: {
          populationSize: 3,
          generations: 2,
          mutationRate: 0.5
        }
      });

      const result = await optimizer.optimize({
        initialCandidate: { 
          program: JSON.stringify({ signature: 'review -> sentiment' }) 
        },
        trainData: [{ review: 'Good', sentiment: 'positive' }],
        valData: [{ review: 'Great', sentiment: 'positive' }],
        componentsToUpdate: ['program'],
        maxMetricCalls: 20
      });

      // Check evolution happened
      const finalProgram = JSON.parse(result.bestCandidate.program);
      expect(finalProgram).toBeDefined();
      expect(result.generation).toBeGreaterThan(0);
    });
  });
});