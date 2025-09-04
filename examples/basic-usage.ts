#!/usr/bin/env tsx
/**
 * Basic usage example - Simple prompt optimization
 * Demonstrates the simplest way to use GEPA for prompt optimization
 */

import { GEPAOptimizer } from '../src/core/optimizer.js';
import { DefaultAdapter } from '../src/adapters/default-adapter.js';
import { OpenAILanguageModel } from '../src/models/openai.js';

// Example: Sentiment Analysis Task
interface SentimentExample {
  text: string;
  sentiment: 'positive' | 'negative' | 'neutral';
}

const SENTIMENT_EXAMPLES: SentimentExample[] = [
  { text: "I love this product! It's amazing!", sentiment: 'positive' },
  { text: "This is terrible. Very disappointed.", sentiment: 'negative' },
  { text: "It's okay, nothing special.", sentiment: 'neutral' },
  { text: "Best purchase I've ever made!", sentiment: 'positive' },
  { text: "Waste of money. Do not buy.", sentiment: 'negative' },
  { text: "Works as expected.", sentiment: 'neutral' },
  { text: "Exceeded all my expectations!", sentiment: 'positive' },
  { text: "Broken on arrival. Awful quality.", sentiment: 'negative' },
  { text: "It does what it says.", sentiment: 'neutral' },
  { text: "Fantastic! Highly recommend!", sentiment: 'positive' },
];

async function main() {
  console.log('🚀 Basic GEPA Usage Example\n');
  console.log('Task: Optimize prompts for sentiment analysis\n');
  
  // Check for API key
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error('❌ Please set OPENAI_API_KEY environment variable');
    console.log('\nExample:');
    console.log('export OPENAI_API_KEY="your-api-key-here"');
    console.log('npm run example:basic\n');
    process.exit(1);
  }
  
  // Initialize language model
  const lm = new OpenAILanguageModel({
    apiKey,
    model: 'gpt-3.5-turbo',
    temperature: 0.3,
    maxTokens: 50
  });
  
  // Create a simple evaluation function
  const evaluator = async (prompt: string, example: SentimentExample): Promise<number> => {
    const fullPrompt = `${prompt}\n\nText: ${example.text}\nSentiment:`;
    const response = await lm.generate(fullPrompt);
    
    // Extract sentiment from response
    const predicted = response.toLowerCase().trim();
    const correct = predicted.includes(example.sentiment);
    
    return correct ? 1.0 : 0.0;
  };
  
  // Create adapter
  const adapter = new DefaultAdapter<SentimentExample>({
    evaluator,
    batchSize: 5
  });
  
  // Initialize optimizer
  const optimizer = new GEPAOptimizer({
    adapter,
    config: {
      populationSize: 4,    // Small population for quick demo
      generations: 3,       // Few generations for quick demo
      mutationRate: 0.3,
      mergeRate: 0.2,
      verbose: true
    }
  });
  
  // Define initial prompt components to optimize
  const initialPrompts = {
    instruction: "Classify the sentiment of the given text as positive, negative, or neutral.",
    
    format: "Output only one word: positive, negative, or neutral.",
    
    examples: `Examples:
- "Great product!" → positive
- "Not worth it." → negative  
- "It's fine." → neutral`
  };
  
  // Split data
  const trainData = SENTIMENT_EXAMPLES.slice(0, 6);
  const valData = SENTIMENT_EXAMPLES.slice(6);
  
  console.log('📊 Dataset:');
  console.log(`- Training examples: ${trainData.length}`);
  console.log(`- Validation examples: ${valData.length}`);
  console.log(`- Components to optimize: ${Object.keys(initialPrompts).join(', ')}\n`);
  
  console.log('🔄 Starting optimization...\n');
  
  // Run optimization
  const startTime = Date.now();
  const result = await optimizer.optimize({
    initialCandidate: initialPrompts,
    trainData,
    valData,
    componentsToUpdate: ['instruction', 'format'] // Don't update examples
  });
  
  const duration = (Date.now() - startTime) / 1000;
  
  // Display results
  console.log('\n' + '='.repeat(50));
  console.log('📈 OPTIMIZATION RESULTS');
  console.log('='.repeat(50));
  
  console.log('\n📊 Metrics:');
  console.log(`- Initial score: ${result.history[0]?.bestScore?.toFixed(3) || 'N/A'}`);
  console.log(`- Final score: ${result.bestScore.toFixed(3)}`);
  console.log(`- Improvement: ${((result.bestScore - (result.history[0]?.bestScore || 0)) * 100).toFixed(1)}%`);
  console.log(`- Generations: ${result.generationsCompleted}`);
  console.log(`- Total evaluations: ${result.totalEvaluations}`);
  console.log(`- Time taken: ${duration.toFixed(1)}s`);
  
  console.log('\n📝 Optimized Prompts:');
  console.log('-'.repeat(50));
  
  for (const [component, text] of Object.entries(result.bestCandidate)) {
    console.log(`\n[${component}]`);
    console.log(text);
  }
  
  // Test the optimized prompt
  console.log('\n' + '='.repeat(50));
  console.log('🧪 TESTING OPTIMIZED PROMPT');
  console.log('='.repeat(50));
  
  const testExamples = [
    { text: "This is absolutely wonderful!", sentiment: 'positive' as const },
    { text: "I hate this so much.", sentiment: 'negative' as const },
    { text: "It's an average product.", sentiment: 'neutral' as const }
  ];
  
  console.log('\nTest results:');
  for (const example of testExamples) {
    const prompt = Object.values(result.bestCandidate).join('\n\n');
    const fullPrompt = `${prompt}\n\nText: ${example.text}\nSentiment:`;
    const response = await lm.generate(fullPrompt);
    const predicted = response.toLowerCase().trim();
    const correct = predicted.includes(example.sentiment);
    
    console.log(`\nText: "${example.text}"`);
    console.log(`Expected: ${example.sentiment}`);
    console.log(`Predicted: ${predicted}`);
    console.log(`Result: ${correct ? '✅ Correct' : '❌ Incorrect'}`);
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('✨ Optimization complete!');
  console.log('='.repeat(50));
  
  // Show how to save results
  console.log('\n💡 To save these results, you could use:');
  console.log('```typescript');
  console.log('import { writeFileSync } from "fs";');
  console.log('writeFileSync("optimized_prompts.json", JSON.stringify(result.bestCandidate, null, 2));');
  console.log('```');
}

// Run the example
main().catch(error => {
  console.error('\n❌ Error:', error);
  console.log('\n💡 Troubleshooting tips:');
  console.log('1. Make sure OPENAI_API_KEY is set correctly');
  console.log('2. Check your OpenAI API quota and rate limits');
  console.log('3. Ensure you have network connectivity');
  process.exit(1);
});