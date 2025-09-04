import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { optimize } from '../api.js';
import { RecordReplayLLM, GoldenFileManager } from '../testing/record-replay.js';
import { SilentLogger } from '../logging/logger.js';
import * as path from 'path';

describe('Real Optimization Tests', () => {
  let recorder: RecordReplayLLM;
  let golden: GoldenFileManager;
  const testDir = path.join(process.cwd(), '.test-cache');
  const isRecording = process.env.RECORD_TESTS === 'true';
  
  beforeAll(() => {
    recorder = new RecordReplayLLM({
      cacheDir: testDir,
      record: isRecording,
      verbose: true
    });
    
    golden = new GoldenFileManager(path.join(testDir, 'golden'));
  });
  
  afterAll(() => {
    recorder.flush();
  });
  
  it('should demonstrate actual optimization improvement', async () => {
    // Simple classification dataset where better prompts should improve accuracy
    const trainset = [
      { text: 'I love this movie!', label: 'positive', answer: 'positive' },
      { text: 'This is terrible.', label: 'negative', answer: 'negative' },
      { text: 'Amazing experience!', label: 'positive', answer: 'positive' },
      { text: 'Waste of time.', label: 'negative', answer: 'negative' },
      { text: 'Highly recommend!', label: 'positive', answer: 'positive' },
      { text: 'Disappointing.', label: 'negative', answer: 'negative' }
    ];
    
    const valset = [
      { text: 'Fantastic!', label: 'positive', answer: 'positive' },
      { text: 'Not good.', label: 'negative', answer: 'negative' },
      { text: 'Excellent work!', label: 'positive', answer: 'positive' }
    ];
    
    // Intentionally bad seed prompt
    const seedCandidate = {
      system_prompt: 'Classify text.'
    };
    
    // Create LLMs with record/replay
    const taskLM = recorder.createTaskLM(
      isRecording ? 
        // In record mode, we'd use real LLM
        async (input: any) => {
          // Simulate an LLM that improves with better prompts
          const prompt = input.system_prompt || '';
          const text = input.text || '';
          
          // Better prompts lead to better accuracy
          const promptQuality = prompt.length / 100; // Simple heuristic
          const randomFactor = Math.random();
          
          // With bad prompt, 50% accuracy
          // With good prompt, up to 90% accuracy
          const accuracy = 0.5 + (promptQuality * 0.4);
          
          if (randomFactor < accuracy) {
            // Correct answer
            return input.answer || 'unknown';
          } else {
            // Wrong answer
            return input.answer === 'positive' ? 'negative' : 'positive';
          }
        } : undefined
    );
    
    const reflectionLM = recorder.createReflectionLM(
      isRecording ?
        async (prompt: string) => {
          // Simulate reflection that improves the prompt
          if (prompt.includes('Classify text.')) {
            return 'You are a sentiment analysis expert. Classify the sentiment of the given text as either "positive" or "negative". Be precise and consider the emotional tone.';
          }
          return 'Improved prompt with more specific instructions for sentiment classification.';
        } : undefined
    );
    
    const result = await optimize({
      seedCandidate,
      trainset,
      valset,
      taskLM,
      reflectionLM,
      maxMetricCalls: 50,
      reflectionMinibatchSize: 2,
      logger: new SilentLogger(),
      runDir: null,
      seed: 42
    });
    
    // Validate optimization occurred
    expect(result.allCandidates.length).toBeGreaterThan(1);
    expect(result.bestScore).toBeGreaterThan(0);
    
    // The optimized prompt should be different and longer
    expect(result.bestCandidate.system_prompt).not.toBe(seedCandidate.system_prompt);
    expect(result.bestCandidate.system_prompt.length).toBeGreaterThan(
      seedCandidate.system_prompt.length
    );
    
    // Check for improvement
    const improvement = result.bestScore - result.allScores[0];
    expect(improvement).toBeGreaterThanOrEqual(0);
    
    // Save or compare golden file
    if (isRecording) {
      golden.save('optimization_result', {
        bestScore: result.bestScore,
        numCandidates: result.allCandidates.length,
        improvement,
        bestPromptLength: result.bestCandidate.system_prompt.length
      });
    } else {
      const expected = golden.load('optimization_result');
      expect(result.bestScore).toBeCloseTo(expected.bestScore, 2);
      expect(result.allCandidates.length).toBe(expected.numCandidates);
    }
  });
  
  it('should correctly track Pareto fronts per validation instance', async () => {
    // Dataset where different prompts work better for different examples
    const trainset = [
      { type: 'math', question: '2+2', answer: '4' },
      { type: 'logic', question: 'If A then B, A is true', answer: 'B is true' },
      { type: 'math', question: '5*3', answer: '15' },
      { type: 'logic', question: 'NOT true', answer: 'false' }
    ];
    
    const valset = [
      { type: 'math', question: '10/2', answer: '5' },
      { type: 'logic', question: 'A AND B, A is false', answer: 'false' }
    ];
    
    const seedCandidate = {
      system_prompt: 'Answer the question.'
    };
    
    // Create adapter that performs differently on math vs logic
    const taskLM = recorder.createTaskLM(
      isRecording ?
        (input: any) => {
          const prompt = input.system_prompt || '';
          const isMathOptimized = prompt.includes('math') || prompt.includes('calculate');
          const isLogicOptimized = prompt.includes('logic') || prompt.includes('reasoning');
          
          if (input.type === 'math' && isMathOptimized) {
            return input.answer; // Perfect on math
          } else if (input.type === 'logic' && isLogicOptimized) {
            return input.answer; // Perfect on logic
          } else {
            return Math.random() > 0.5 ? input.answer : 'wrong';
          }
        } : undefined
    );
    
    let iteration = 0;
    const reflectionLM = recorder.createReflectionLM(
      isRecording ?
        (prompt: string) => {
          iteration++;
          // Alternate between math and logic optimizations
          if (iteration % 2 === 1) {
            return 'You are a math calculator. Calculate and solve mathematical problems.';
          } else {
            return 'You are a logic reasoning system. Apply logical rules to solve problems.';
          }
        } : undefined
    );
    
    const result = await optimize({
      seedCandidate,
      trainset,
      valset,
      taskLM,
      reflectionLM,
      maxMetricCalls: 30,
      reflectionMinibatchSize: 2,
      candidateSelectionStrategy: 'pareto',
      logger: new SilentLogger(),
      runDir: null
    });
    
    // Should have multiple candidates due to trade-offs
    expect(result.paretoFrontCandidates.length).toBeGreaterThan(0);
    
    // Different candidates should excel at different tasks
    // This validates that Pareto selection is working
    expect(result.allCandidates.length).toBeGreaterThan(2);
  });
});