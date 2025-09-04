/**
 * Tests for all adapter implementations
 */

import { describe, it, expect, beforeEach } from '@jest/globals';
import { 
  DefaultAdapter,
  DSPyAdapter,
  DSPyFullProgramAdapter,
  TerminalBenchAdapter,
  AnyMathsAdapter
} from '../../src/adapters/index.js';
import { TestLLM } from '../infrastructure/record-replay.js';
import { ComponentMap } from '../../src/types/index.js';

describe('Adapters', () => {
  let testLLM: TestLLM;
  
  beforeEach(() => {
    testLLM = new TestLLM({ mode: 'replay' });
  });
  
  describe('DefaultAdapter', () => {
    it('should evaluate batches correctly', async () => {
      const evaluator = async (prompt: string, example: any): Promise<number> => {
        return example.score || 0.5;
      };
      
      const adapter = new DefaultAdapter({
        evaluator,
        batchSize: 2
      });
      
      const batch = [
        { input: 'test1', score: 0.7 },
        { input: 'test2', score: 0.3 },
        { input: 'test3', score: 0.9 }
      ];
      
      const candidate = { prompt: 'Test prompt' };
      
      const result = await adapter.evaluate(batch, candidate, false);
      
      expect(result.scores).toEqual([0.7, 0.3, 0.9]);
      expect(result.outputs).toHaveLength(3);
      expect(result.trajectories).toBeNull();
    });
    
    it('should capture traces when requested', async () => {
      const evaluator = async (prompt: string, example: any): Promise<number> => {
        return 0.5;
      };
      
      const adapter = new DefaultAdapter({
        evaluator,
        batchSize: 1,
        traceCapture: async (prompt, example) => ({
          prompt,
          example,
          timestamp: Date.now()
        })
      });
      
      const batch = [{ input: 'test' }];
      const candidate = { prompt: 'Test' };
      
      const result = await adapter.evaluate(batch, candidate, true);
      
      expect(result.trajectories).not.toBeNull();
      expect(result.trajectories).toHaveLength(1);
      expect(result.trajectories![0]).toHaveProperty('prompt');
    });
    
    it('should create reflective dataset', async () => {
      const adapter = new DefaultAdapter({
        evaluator: async () => 0.5,
        batchSize: 1
      });
      
      const evalBatch = {
        outputs: [{ result: 'output1' }, { result: 'output2' }],
        scores: [0.8, 0.3],
        trajectories: null
      };
      
      const candidate = {
        component1: 'Text 1',
        component2: 'Text 2'
      };
      
      const dataset = await adapter.makeReflectiveDataset(
        candidate,
        evalBatch,
        ['component1']
      );
      
      expect(dataset).toHaveProperty('component1');
      expect(dataset.component1).toBeInstanceOf(Array);
      expect(dataset.component1[0]).toHaveProperty('Score');
      expect(dataset.component1[0]).toHaveProperty('Output');
    });
  });
  
  describe('DSPyAdapter', () => {
    it('should handle DSPy signatures correctly', async () => {
      const adapter = new DSPyAdapter({
        lm: async (prompt: string) => testLLM.generate(prompt),
        modules: {
          classifier: {
            type: 'Predict',
            signature: {
              inputs: ['text'],
              outputs: ['category'],
              instruction: 'Classify the text'
            }
          }
        }
      });
      
      const batch = [
        { text: 'positive review', category: 'positive' },
        { text: 'negative review', category: 'negative' }
      ];
      
      const candidate = {
        'classifier.instruction': 'Classify sentiment of the text'
      };
      
      const result = await adapter.evaluate(batch, candidate, false);
      
      expect(result.scores).toHaveLength(2);
      expect(result.outputs).toHaveLength(2);
    });
    
    it('should support ChainOfThought module', async () => {
      const adapter = new DSPyAdapter({
        lm: async (prompt: string) => {
          if (prompt.includes('think step by step')) {
            return 'Step 1: Analyze\nStep 2: Process\nAnswer: result';
          }
          return 'Direct answer';
        },
        modules: {
          reasoner: {
            type: 'ChainOfThought',
            signature: {
              inputs: ['question'],
              outputs: ['reasoning', 'answer'],
              instruction: 'Answer the question'
            }
          }
        }
      });
      
      const batch = [
        { question: 'What is 2+2?', answer: '4' }
      ];
      
      const candidate = {
        'reasoner.instruction': 'Think step by step to answer'
      };
      
      const result = await adapter.evaluate(batch, candidate, true);
      
      expect(result.scores).toHaveLength(1);
      expect(result.trajectories).not.toBeNull();
      
      if (result.trajectories) {
        const trace = result.trajectories[0] as any;
        expect(trace).toHaveProperty('reasoning');
      }
    });
    
    it('should create DSPy-specific reflective dataset', async () => {
      const adapter = new DSPyAdapter({
        lm: async () => 'test output',
        modules: {
          test: {
            type: 'Predict',
            signature: {
              inputs: ['input'],
              outputs: ['output'],
              instruction: 'Process input'
            }
          }
        }
      });
      
      const evalBatch = {
        outputs: [
          { output: 'generated1', reasoning: 'step1' },
          { output: 'generated2', reasoning: 'step2' }
        ],
        scores: [0.9, 0.4],
        trajectories: [
          { moduleType: 'Predict', input: 'in1', output: 'out1' },
          { moduleType: 'Predict', input: 'in2', output: 'out2' }
        ]
      };
      
      const candidate = {
        'test.instruction': 'Current instruction'
      };
      
      const dataset = await adapter.makeReflectiveDataset(
        candidate,
        evalBatch,
        ['test.instruction']
      );
      
      expect(dataset).toHaveProperty('test.instruction');
      const examples = dataset['test.instruction'];
      expect(examples).toBeInstanceOf(Array);
      expect(examples[0]).toHaveProperty('Score');
      expect(examples[0]).toHaveProperty('Module Type');
    });
  });
  
  describe('DSPyFullProgramAdapter', () => {
    it('should evolve DSPy program structure', async () => {
      const adapter = new DSPyFullProgramAdapter({
        lm: async (prompt: string) => {
          if (prompt.includes('major restructure')) {
            return JSON.stringify({
              modules: [
                { name: 'new_module', type: 'ChainOfThought' }
              ],
              flow: 'new_module'
            });
          }
          return 'Default response';
        },
        allowedModuleTypes: ['Predict', 'ChainOfThought'],
        maxModules: 3,
        evolutionStrategy: 'majorRestructure'
      });
      
      const batch = [{ input: 'test', output: 'expected' }];
      
      const candidate = {
        'program.structure': JSON.stringify({
          modules: [{ name: 'old', type: 'Predict' }],
          flow: 'old'
        })
      };
      
      const result = await adapter.evaluate(batch, candidate, false);
      
      expect(result.scores).toHaveLength(1);
      expect(result.outputs).toHaveLength(1);
    });
    
    it('should propose new program structures', () => {
      const adapter = new DSPyFullProgramAdapter({
        lm: async () => 'response',
        evolutionStrategy: 'incrementalImprove'
      });
      
      const candidate = {
        'program.structure': JSON.stringify({
          modules: [
            { name: 'understand', type: 'Predict' },
            { name: 'solve', type: 'ChainOfThought' }
          ],
          flow: 'understand -> solve'
        })
      };
      
      const reflectiveDataset = {
        'program.structure': [
          {
            Score: 0.3,
            Feedback: 'Poor reasoning',
            'Module Performance': { understand: 0.5, solve: 0.2 }
          }
        ]
      };
      
      const newTexts = adapter.proposeNewTexts(
        candidate,
        reflectiveDataset,
        ['program.structure']
      );
      
      expect(newTexts).toHaveProperty('program.structure');
      
      // Should modify the program based on feedback
      const newProgram = JSON.parse(newTexts['program.structure']);
      expect(newProgram).toHaveProperty('modules');
      expect(newProgram).toHaveProperty('flow');
    });
  });
  
  describe('TerminalBenchAdapter', () => {
    it('should evaluate terminal command sequences', async () => {
      const mockExecutor = async (
        instruction: string,
        task: string,
        initialState: any
      ) => ({
        commands: [
          {
            command: 'ls',
            stdout: 'file1.txt file2.txt',
            stderr: '',
            exitCode: 0,
            timestamp: Date.now()
          },
          {
            command: 'cat file1.txt',
            stdout: 'content',
            stderr: '',
            exitCode: 0,
            timestamp: Date.now()
          }
        ],
        finalState: {
          cwd: '.',
          files: { 'file1.txt': 'content' }
        },
        success: true
      });
      
      const adapter = new TerminalBenchAdapter({
        agentExecutor: mockExecutor,
        maxCommands: 10,
        timeout: 5000
      });
      
      const batch = [
        {
          task: 'Read file1.txt',
          initialState: { cwd: '.', files: { 'file1.txt': 'content' } },
          expectedOutcome: { files: { 'file1.txt': 'content' } }
        }
      ];
      
      const candidate = {
        terminal_instruction: 'Complete the task: {task}'
      };
      
      const result = await adapter.evaluate(batch, candidate, true);
      
      expect(result.scores).toHaveLength(1);
      expect(result.scores[0]).toBeGreaterThan(0); // Should have some success
      expect(result.outputs[0]).toHaveProperty('commandCount');
      expect(result.outputs[0].commandCount).toBe(2);
      expect(result.trajectories).not.toBeNull();
    });
    
    it('should create terminal-specific reflective dataset', async () => {
      const adapter = new TerminalBenchAdapter({
        agentExecutor: async () => ({
          commands: [],
          finalState: { cwd: '.', files: {} },
          success: false
        }),
        verbose: false
      });
      
      const evalBatch = {
        outputs: [
          { finalState: {}, success: true, commandCount: 5 },
          { finalState: {}, success: false, commandCount: 0 }
        ],
        scores: [0.8, 0.0],
        trajectories: [
          {
            commands: [
              { command: 'ls', exitCode: 0, stdout: '', stderr: '', timestamp: 0 }
            ],
            finalState: { cwd: '.', files: {} },
            success: true
          },
          {
            commands: [],
            finalState: { cwd: '.', files: {} },
            success: false
          }
        ]
      };
      
      const candidate = {
        terminal_instruction: 'Execute commands'
      };
      
      const dataset = await adapter.makeReflectiveDataset(
        candidate,
        evalBatch,
        ['terminal_instruction']
      );
      
      expect(dataset).toHaveProperty('terminal_instruction');
      const examples = dataset.terminal_instruction;
      expect(examples[0]).toHaveProperty('Score');
      expect(examples[0]).toHaveProperty('Commands Used');
      expect(examples[0]).toHaveProperty('Feedback');
    });
  });
  
  describe('AnyMathsAdapter', () => {
    it('should solve math problems', async () => {
      const adapter = new AnyMathsAdapter({
        model: async (prompt: string) => {
          if (prompt.includes('2+2')) {
            return 'Step 1: Add 2 and 2\nStep 2: 2 + 2 = 4\nAnswer: 4';
          }
          return 'Answer: 0';
        },
        requireSteps: true
      });
      
      const batch = [
        { problem: 'What is 2+2?', answer: '4' },
        { problem: 'What is 5*5?', answer: '25' }
      ];
      
      const candidate = {
        math_prompt: 'Solve the problem step by step'
      };
      
      const result = await adapter.evaluate(batch, candidate, true);
      
      expect(result.scores).toHaveLength(2);
      expect(result.scores[0]).toBe(1.0); // Correct answer
      expect(result.scores[1]).toBe(0.0); // Wrong answer
      
      expect(result.outputs[0]).toHaveProperty('answer');
      expect(result.outputs[0]).toHaveProperty('reasoning');
      expect(result.outputs[0]).toHaveProperty('steps');
      expect(result.outputs[0]).toHaveProperty('confidence');
    });
    
    it('should handle different answer formats', async () => {
      const adapter = new AnyMathsAdapter({
        model: async (prompt: string) => {
          if (prompt.includes('fraction')) {
            return 'Answer: 1/2';
          }
          if (prompt.includes('decimal')) {
            return 'The result is 0.5';
          }
          if (prompt.includes('boxed')) {
            return '\\boxed{42}';
          }
          return '### 99';
        }
      });
      
      const batch = [
        { problem: 'fraction problem', answer: '0.5' },
        { problem: 'decimal problem', answer: '1/2' },
        { problem: 'boxed problem', answer: '42' },
        { problem: 'hash problem', answer: '99' }
      ];
      
      const candidate = { math_prompt: 'Solve' };
      
      const result = await adapter.evaluate(batch, candidate, false);
      
      // Check answer extraction and comparison
      expect(result.scores[0]).toBe(1.0); // 1/2 = 0.5
      expect(result.scores[1]).toBe(1.0); // 0.5 = 1/2
      expect(result.scores[2]).toBe(1.0); // boxed 42
      expect(result.scores[3]).toBe(1.0); // ### format
    });
    
    it('should create math-specific reflective dataset', async () => {
      const adapter = new AnyMathsAdapter({
        model: async () => 'Answer: 5'
      });
      
      const evalBatch = {
        outputs: [
          { 
            answer: '5', 
            reasoning: 'Added numbers',
            steps: ['Step 1', 'Step 2'],
            confidence: 0.8
          },
          {
            answer: '10',
            reasoning: 'Wrong calculation',
            steps: [],
            confidence: 0.3
          }
        ],
        scores: [1.0, 0.0],
        trajectories: [
          {
            problem: '2+3=?',
            steps: [],
            finalAnswer: '5',
            isCorrect: true
          },
          {
            problem: '5+5=?',
            steps: [],
            finalAnswer: '10',
            isCorrect: false,
            errors: ['Calculation error']
          }
        ]
      };
      
      const candidate = {
        math_prompt: 'Solve math problems'
      };
      
      const dataset = await adapter.makeReflectiveDataset(
        candidate,
        evalBatch,
        ['math_prompt']
      );
      
      expect(dataset).toHaveProperty('math_prompt');
      const examples = dataset.math_prompt;
      expect(examples[0]).toHaveProperty('Score');
      expect(examples[0]).toHaveProperty('Problem');
      expect(examples[0]).toHaveProperty('Generated Answer');
      expect(examples[0]).toHaveProperty('Feedback');
      expect(examples[0]).toHaveProperty('Improvements');
    });
  });
});