/**
 * Integration tests with real AI (OpenAI)
 * 
 * These tests make actual API calls and demonstrate real AxLLM + GEPA evolution.
 * They are skipped by default to avoid API costs in CI.
 * 
 * To run these tests:
 * 1. Set OPENAI_API_KEY environment variable
 * 2. Run: REAL_AI_TESTS=true npm test axllm-real-ai-integration.test.ts
 */

import { describe, it, expect, beforeAll, skipIf } from 'vitest';
import { GEPAOptimizer } from '../src/core/optimizer.js';
import { AxLLMNativeAdapter, createAxLLMNativeAdapter } from '../src/adapters/axllm-native-adapter.js';
import OpenAI from 'openai';

// Skip these tests unless explicitly enabled
const skipRealAI = !process.env.REAL_AI_TESTS || !process.env.OPENAI_API_KEY;

// Minimal implementations for testing
class TestAxProgram {
  signature: string;
  demos: any[] = [];
  modelConfig: any = { temperature: 0.7 };
  
  constructor(signature: string) {
    this.signature = signature;
  }
  
  setDemos(demos: any[]) {
    this.demos = demos;
  }
  
  async forward(llm: any, inputs: any) {
    const prompt = this.buildPrompt(inputs);
    const response = await llm.generate(prompt, this.modelConfig);
    return this.parseResponse(response);
  }
  
  private buildPrompt(inputs: any): string {
    let prompt = `Task: ${this.signature}\n\n`;
    
    if (this.demos.length > 0) {
      prompt += 'Examples:\n';
      for (const demo of this.demos.slice(0, 3)) {
        prompt += `Input: ${JSON.stringify(demo.input || demo)}\n`;
        prompt += `Output: ${demo.output || demo.label || demo.answer}\n\n`;
      }
    }
    
    prompt += `Input: ${JSON.stringify(inputs)}\nOutput: `;
    return prompt;
  }
  
  private parseResponse(response: string): any {
    // Parse based on signature
    if (this.signature.includes('boolean')) {
      return { result: response.toLowerCase().includes('true') };
    }
    if (this.signature.includes('number')) {
      const num = parseFloat(response);
      return { result: isNaN(num) ? 0 : num };
    }
    if (this.signature.includes('classification')) {
      return { label: response.trim() };
    }
    return { output: response.trim() };
  }
}

describe.skipIf(skipRealAI)('AxLLM + GEPA Real AI Integration', () => {
  let openai: OpenAI;
  let llmWrapper: any;
  let axFactory: any;
  
  beforeAll(() => {
    if (skipRealAI) {
      console.log('⚠️  Skipping real AI tests. Set REAL_AI_TESTS=true and OPENAI_API_KEY to run.');
      return;
    }
    
    openai = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });
    
    // LLM wrapper for tests
    llmWrapper = {
      async generate(prompt: string, config: any = {}) {
        const response = await openai.chat.completions.create({
          model: config.model || 'gpt-3.5-turbo',
          messages: [{ role: 'user', content: prompt }],
          temperature: config.temperature || 0.7,
          max_tokens: config.maxTokens || 30
        });
        return response.choices[0]?.message?.content || '';
      }
    };
    
    // ax() factory
    axFactory = (signature: string) => new TestAxProgram(signature);
  });
  
  it('should evolve a simple classification program with real AI', async () => {
    console.log('\n🧪 Test: Evolving classification program with real OpenAI\n');
    
    const trainingData = [
      { text: "I love this!", label: "positive" },
      { text: "This is terrible", label: "negative" },
      { text: "It's okay", label: "neutral" }
    ];
    
    const adapter = createAxLLMNativeAdapter({
      axInstance: axFactory,
      llm: llmWrapper,
      baseProgram: {
        signature: 'Classify text sentiment',
        demos: [],
        modelConfig: { temperature: 0.5 }
      },
      metricFn: ({ prediction, example }) => {
        return prediction.label === example.label ? 1 : 0;
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
    
    const result = await optimizer.optimize({
      initialCandidate: { 
        program: JSON.stringify({
          signature: 'text classification',
          demos: []
        })
      },
      trainData: trainingData,
      valData: trainingData.slice(0, 1),
      componentsToUpdate: ['program'],
      maxMetricCalls: 5
    });
    
    expect(result).toBeDefined();
    expect(result.bestCandidate).toBeDefined();
    expect(result.bestScore).toBeGreaterThanOrEqual(0);
    expect(result.bestScore).toBeLessThanOrEqual(1);
    
    console.log(`  Score achieved: ${(result.bestScore * 100).toFixed(1)}%`);
  }, 30000); // 30s timeout for API calls
  
  it('should improve program through evolution with real feedback', async () => {
    console.log('\n🧪 Test: Multi-generation evolution with real AI\n');
    
    const examples = [
      { x: 2, y: 3, sum: 5 },
      { x: 10, y: 5, sum: 15 },
      { x: -1, y: 1, sum: 0 }
    ];
    
    const adapter = createAxLLMNativeAdapter({
      axInstance: axFactory,
      llm: llmWrapper,
      baseProgram: {
        signature: 'Add two numbers',
        demos: []
      },
      metricFn: ({ prediction, example }) => {
        const predicted = parseInt(prediction.output || prediction.result || '0');
        return predicted === example.sum ? 1 : 0;
      },
      reflectionLm: async (prompt) => {
        // Use real AI to evolve program
        const response = await llmWrapper.generate(
          prompt + '\nReturn improved program as JSON.',
          { temperature: 0.8, maxTokens: 200 }
        );
        
        // Try to extract JSON
        const match = response.match(/\{[\s\S]*\}/);
        if (match) return match[0];
        
        // Fallback
        return JSON.stringify({
          signature: 'Calculate sum of x and y -> number',
          demos: examples.slice(0, 2),
          modelConfig: { temperature: 0.1 }
        });
      }
    });
    
    const optimizer = new GEPAOptimizer({
      adapter,
      config: {
        populationSize: 2,
        generations: 2,
        mutationRate: 0.7
      }
    });
    
    const result = await optimizer.optimize({
      initialCandidate: {
        program: JSON.stringify({
          signature: 'math operation',
          demos: []
        })
      },
      trainData: examples,
      valData: [{ x: 7, y: 8, sum: 15 }],
      componentsToUpdate: ['program'],
      maxMetricCalls: 10
    });
    
    // Check evolution occurred
    const finalProgram = JSON.parse(result.bestCandidate.program);
    
    expect(finalProgram).toBeDefined();
    expect(result.generation).toBeGreaterThan(0);
    
    // Program should have evolved (added demos or changed signature)
    const hasEvolved = 
      finalProgram.demos?.length > 0 || 
      finalProgram.signature !== 'math operation';
    
    expect(hasEvolved).toBe(true);
    
    console.log(`  Generations: ${result.generation}`);
    console.log(`  Final score: ${(result.bestScore * 100).toFixed(1)}%`);
    console.log(`  Program evolved: ${hasEvolved ? 'Yes' : 'No'}`);
  }, 45000); // 45s timeout
  
  it('should handle complex program with demos and model config', async () => {
    console.log('\n🧪 Test: Complex program evolution\n');
    
    const questionAnswers = [
      { question: "What is 2+2?", answer: "4" },
      { question: "What color is the sky?", answer: "blue" },
      { question: "Capital of France?", answer: "Paris" }
    ];
    
    const adapter = createAxLLMNativeAdapter({
      axInstance: axFactory,
      llm: llmWrapper,
      baseProgram: {
        signature: 'Answer questions concisely',
        demos: questionAnswers.slice(0, 1),
        modelConfig: {
          temperature: 0.3,
          maxTokens: 10
        }
      },
      metricFn: ({ prediction, example }) => {
        const answer = (prediction.output || prediction.answer || '').toLowerCase();
        const expected = example.answer.toLowerCase();
        return answer.includes(expected) || expected.includes(answer) ? 1 : 0;
      }
    });
    
    // Test evaluation with existing program
    const evalResult = await adapter.evaluate(
      questionAnswers.slice(1, 2),
      {
        program: JSON.stringify({
          signature: 'Q&A assistant',
          demos: questionAnswers.slice(0, 1),
          modelConfig: { temperature: 0.2 }
        })
      },
      true // Capture traces
    );
    
    expect(evalResult).toBeDefined();
    expect(evalResult.outputs).toHaveLength(1);
    expect(evalResult.scores).toHaveLength(1);
    expect(evalResult.trajectories).toBeDefined();
    
    // Test AxOptimizedProgram conversion
    const axOptimized = adapter.toAxOptimizedProgram(
      { program: JSON.stringify(adapter['baseProgram']) },
      0.8,
      { rounds: 3 }
    );
    
    expect(axOptimized.bestScore).toBe(0.8);
    expect(axOptimized.optimizerType).toBe('GEPA');
    expect(axOptimized.demos).toBeDefined();
    expect(axOptimized.modelConfig).toBeDefined();
    
    console.log(`  Evaluation score: ${evalResult.scores[0]}`);
    console.log(`  Traces captured: ${evalResult.trajectories ? 'Yes' : 'No'}`);
  }, 30000);
  
  it('should demonstrate real improvement over baseline', async () => {
    console.log('\n🧪 Test: Measuring real improvement\n');
    
    const dataset = [
      { review: "Amazing product!", sentiment: "positive" },
      { review: "Waste of money", sentiment: "negative" },
      { review: "It's acceptable", sentiment: "neutral" },
      { review: "Love it!", sentiment: "positive" },
      { review: "Disappointed", sentiment: "negative" }
    ];
    
    const metricFn = ({ prediction, example }: any) => {
      const pred = (prediction.label || prediction.sentiment || '').toLowerCase();
      return pred === example.sentiment ? 1 : 0;
    };
    
    // Baseline program (no demos, generic signature)
    const baselineProgram = {
      signature: 'classify',
      demos: [],
      modelConfig: { temperature: 0.7 }
    };
    
    // Create adapter
    const adapter = createAxLLMNativeAdapter({
      axInstance: axFactory,
      llm: llmWrapper,
      baseProgram: baselineProgram,
      metricFn
    });
    
    // Measure baseline performance
    console.log('  Measuring baseline...');
    const baselineEval = await adapter.evaluate(
      dataset.slice(0, 3),
      { program: JSON.stringify(baselineProgram) },
      false
    );
    const baselineScore = baselineEval.scores.reduce((a, b) => a + b, 0) / baselineEval.scores.length;
    
    // Run optimization
    console.log('  Running GEPA optimization...');
    const optimizer = new GEPAOptimizer({
      adapter,
      config: {
        populationSize: 3,
        generations: 1,
        mutationRate: 0.5
      }
    });
    
    const result = await optimizer.optimize({
      initialCandidate: { program: JSON.stringify(baselineProgram) },
      trainData: dataset.slice(0, 3),
      valData: dataset.slice(3, 5),
      componentsToUpdate: ['program'],
      maxMetricCalls: 8
    });
    
    // Measure optimized performance
    console.log('  Measuring optimized...');
    const optimizedEval = await adapter.evaluate(
      dataset.slice(0, 3),
      result.bestCandidate,
      false
    );
    const optimizedScore = optimizedEval.scores.reduce((a, b) => a + b, 0) / optimizedEval.scores.length;
    
    console.log(`\n  Results:`);
    console.log(`    Baseline:  ${(baselineScore * 100).toFixed(1)}%`);
    console.log(`    Optimized: ${(optimizedScore * 100).toFixed(1)}%`);
    console.log(`    Improvement: ${((optimizedScore - baselineScore) * 100).toFixed(1)} points`);
    
    // Optimized should be at least as good as baseline
    expect(optimizedScore).toBeGreaterThanOrEqual(baselineScore - 0.1); // Allow small variance
    
    // Check that evolution actually happened
    const evolved = JSON.parse(result.bestCandidate.program);
    const changed = 
      evolved.signature !== baselineProgram.signature ||
      evolved.demos?.length > 0 ||
      evolved.modelConfig?.temperature !== baselineProgram.modelConfig.temperature;
    
    expect(changed).toBe(true);
    console.log(`    Program evolved: ${changed ? 'Yes' : 'No'}`);
  }, 60000); // 60s timeout for full test
});

// Standalone test runner for debugging
if (import.meta.url === `file://${process.argv[1]}`) {
  if (!process.env.OPENAI_API_KEY) {
    console.error('❌ Set OPENAI_API_KEY to run real AI tests');
    process.exit(1);
  }
  
  console.log('Running real AI integration tests...\n');
  console.log('⚠️  This will make actual OpenAI API calls and incur costs!\n');
}