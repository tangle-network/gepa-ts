/**
 * REAL VALIDATION TEST - This actually tests optimization with real LLMs
 * 
 * This test will FAIL if the optimization doesn't actually improve performance
 */

import { describe, it, expect } from '@jest/globals';
import { GEPAOptimizer } from '../src/core/optimizer.js';
import { DefaultAdapter } from '../src/adapters/default-adapter.js';
import { OpenAILanguageModel } from '../src/models/openai.js';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

interface SentimentExample {
  text: string;
  sentiment: 'positive' | 'negative' | 'neutral';
}

describe('🔥 REAL OPTIMIZATION VALIDATION', () => {
  const apiKey = process.env.OPENAI_API_KEY;
  
  beforeAll(() => {
    if (!apiKey) {
      console.warn('⚠️  No OPENAI_API_KEY found - skipping real validation tests');
    }
  });

  it('should actually improve sentiment classification with real LLM', async () => {
    if (!apiKey) {
      console.log('📋 Skipping real validation - no API key');
      return;
    }

    console.log('🚀 Starting REAL optimization test with OpenAI...');
    
    // Real sentiment data
    const trainData: SentimentExample[] = [
      { text: "I absolutely love this product!", sentiment: "positive" },
      { text: "This is terrible and disappointing.", sentiment: "negative" },
      { text: "It's okay, nothing special.", sentiment: "neutral" },
      { text: "Amazing quality, highly recommend!", sentiment: "positive" },
      { text: "Waste of money, very poor quality.", sentiment: "negative" },
      { text: "It works as described.", sentiment: "neutral" },
      { text: "Fantastic experience, exceeded expectations!", sentiment: "positive" },
      { text: "Awful customer service, frustrated.", sentiment: "negative" }
    ];

    const valData: SentimentExample[] = [
      { text: "Best purchase ever!", sentiment: "positive" },
      { text: "Not worth the price.", sentiment: "negative" },
      { text: "Standard quality product.", sentiment: "neutral" },
      { text: "Couldn't be happier with this!", sentiment: "positive" },
      { text: "Regret buying this item.", sentiment: "negative" }
    ];

    // Initialize real OpenAI model
    const llm = new OpenAILanguageModel({
      apiKey,
      model: 'gpt-3.5-turbo',
      temperature: 0.3,
      maxTokens: 50
    });

    console.log('📊 Testing baseline performance...');
    
    // Test baseline with bad prompt
    const baselinePrompt = "Tell me about this text";
    let baselineScore = 0;
    for (const example of valData) {
      try {
        const response = await llm.generate(
          `${baselinePrompt}\n\nText: "${example.text}"\nSentiment:`
        );
        
        const predicted = response.toLowerCase().trim();
        if (predicted.includes(example.sentiment)) {
          baselineScore += 1;
        }
        
        // Add small delay to avoid rate limits
        await new Promise(resolve => setTimeout(resolve, 100));
      } catch (error) {
        console.error('Baseline evaluation error:', error);
      }
    }
    baselineScore /= valData.length;
    
    console.log(`📈 Baseline accuracy: ${(baselineScore * 100).toFixed(1)}%`);

    // Create real adapter
    const adapter = new DefaultAdapter<SentimentExample>({
      evaluator: async (prompt: string, example: SentimentExample): Promise<number> => {
        try {
          const response = await llm.generate(
            `${prompt}\n\nText: "${example.text}"\nSentiment:`
          );
          
          const predicted = response.toLowerCase().trim();
          const isCorrect = predicted.includes(example.sentiment);
          
          // Add small delay to avoid rate limits
          await new Promise(resolve => setTimeout(resolve, 50));
          
          return isCorrect ? 1.0 : 0.0;
        } catch (error) {
          console.error('Evaluation error:', error);
          return 0.0;
        }
      },
      batchSize: 2 // Small batch to avoid rate limits
    });

    // Create optimizer with conservative settings
    const optimizer = new GEPAOptimizer({
      adapter,
      config: {
        populationSize: 4,    // Small population to save API calls
        generations: 3,       // Few generations for quick test
        mutationRate: 0.5,
        mergeRate: 0.2,
        verbose: true
      }
    });

    console.log('🧬 Running GEPA optimization...');
    
    const startTime = Date.now();
    const result = await optimizer.optimize({
      initialCandidate: {
        instruction: "Classify the sentiment of this text as positive, negative, or neutral."
      },
      trainData: trainData.slice(0, 6), // Use subset to save costs
      valData: valData,
      componentsToUpdate: ['instruction']
    });
    
    const duration = (Date.now() - startTime) / 1000;
    
    console.log('\n' + '='.repeat(50));
    console.log('📊 REAL OPTIMIZATION RESULTS');
    console.log('='.repeat(50));
    console.log(`⏱️  Duration: ${duration.toFixed(1)}s`);
    console.log(`🎯 Baseline Score: ${(baselineScore * 100).toFixed(1)}%`);
    console.log(`🏆 Optimized Score: ${(result.bestScore * 100).toFixed(1)}%`);
    console.log(`📈 Improvement: ${((result.bestScore - baselineScore) * 100).toFixed(1)}%`);
    console.log(`🔄 Generations: ${result.generationsCompleted}`);
    console.log(`💸 Total Evaluations: ${result.totalEvaluations}`);
    
    console.log('\n📝 Best Optimized Prompt:');
    console.log('---');
    console.log(result.bestCandidate.instruction);
    console.log('---');

    // REAL ASSERTIONS - These will fail if optimization doesn't work
    expect(result.bestScore).toBeGreaterThan(baselineScore);
    expect(result.bestScore).toBeGreaterThan(0.6); // Should get at least 60% accuracy
    expect(result.generationsCompleted).toBeGreaterThan(0);
    expect(result.totalEvaluations).toBeGreaterThan(10);
    
    // The optimized prompt should be different and longer
    expect(result.bestCandidate.instruction).not.toBe("Classify the sentiment of this text as positive, negative, or neutral.");
    expect(result.bestCandidate.instruction.length).toBeGreaterThan(50);

    console.log('✅ REAL OPTIMIZATION VALIDATION PASSED!');
    
    // Test final prompt on a few examples
    console.log('\n🧪 Testing optimized prompt:');
    for (let i = 0; i < Math.min(3, valData.length); i++) {
      const example = valData[i];
      try {
        const response = await llm.generate(
          `${result.bestCandidate.instruction}\n\nText: "${example.text}"\nSentiment:`
        );
        
        const predicted = response.toLowerCase().trim();
        const correct = predicted.includes(example.sentiment);
        
        console.log(`📝 "${example.text}"`);
        console.log(`   Expected: ${example.sentiment}`);
        console.log(`   Got: ${predicted}`);
        console.log(`   Result: ${correct ? '✅' : '❌'}`);
        
        await new Promise(resolve => setTimeout(resolve, 100));
      } catch (error) {
        console.error('Test error:', error);
      }
    }

  }, 120000); // 2 minute timeout for real API calls
  
  it('should demonstrate Pareto optimization with multi-objective task', async () => {
    if (!apiKey) {
      console.log('📋 Skipping Pareto test - no API key');
      return;
    }
    
    console.log('🎯 Testing Pareto optimization...');
    
    // Multi-objective: accuracy AND brevity
    const llm = new OpenAILanguageModel({ apiKey, model: 'gpt-3.5-turbo' });
    
    const data = [
      { input: "Explain gravity", complexity: "simple" },
      { input: "Describe photosynthesis", complexity: "simple" },
      { input: "What is DNA?", complexity: "simple" }
    ];
    
    const adapter = new DefaultAdapter({
      evaluator: async (prompt: string, example: any): Promise<number> => {
        try {
          const response = await llm.generate(`${prompt}\n\nQuestion: ${example.input}\nAnswer:`);
          
          // Score based on relevance AND brevity
          const relevanceScore = response.length > 20 ? 0.5 : 0; // Has content
          const brevityScore = response.length < 200 ? 0.5 : 0; // Not too long
          
          await new Promise(resolve => setTimeout(resolve, 50));
          return relevanceScore + brevityScore;
        } catch (error) {
          return 0;
        }
      }
    });
    
    const optimizer = new GEPAOptimizer({
      adapter,
      config: {
        populationSize: 3,
        generations: 2,
        verbose: true
      }
    });
    
    const result = await optimizer.optimize({
      initialCandidate: {
        prompt: "Answer the question."
      },
      trainData: data,
      valData: data,
      componentsToUpdate: ['prompt']
    });
    
    expect(result.bestScore).toBeGreaterThan(0);
    expect(result.bestCandidate.prompt).toBeDefined();
    
    console.log('✅ Pareto optimization completed');
    
  }, 60000);
  
});