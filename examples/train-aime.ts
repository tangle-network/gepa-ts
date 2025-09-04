#!/usr/bin/env tsx
/**
 * AIME training example - Mathematical reasoning optimization
 * Demonstrates optimizing prompts for AIME-style mathematical problems
 */

import { GEPAOptimizer } from '../src/core/optimizer.js';
import { AnyMathsAdapter } from '../src/adapters/anymaths-adapter.js';
import { loadAIME2025, loadValidationSplit } from '../src/datasets/aime.js';
import { OpenAILanguageModel } from '../src/models/openai.js';
import { WandBIntegration } from '../src/integrations/wandb.js';
import { writeFileSync } from 'fs';
import { join } from 'path';

async function main() {
  console.log('🔬 Starting AIME Mathematical Reasoning Optimization');
  
  // Load datasets
  console.log('📚 Loading AIME datasets...');
  const trainProblems = await loadAIME2025();
  const valProblems = await loadValidationSplit();
  
  console.log(`Loaded ${trainProblems.length} training problems`);
  console.log(`Loaded ${valProblems.length} validation problems`);
  
  // Initialize language model
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error('❌ OPENAI_API_KEY environment variable not set');
    process.exit(1);
  }
  
  const lm = new OpenAILanguageModel({
    apiKey,
    model: 'gpt-4-turbo-preview',
    temperature: 0.7,
    maxTokens: 2000
  });
  
  // Create adapter
  const adapter = new AnyMathsAdapter({
    model: async (prompt: string) => lm.generate(prompt),
    requireSteps: true,
    timeout: 60000
  });
  
  // Initialize WandB (optional)
  let wandb: WandBIntegration | undefined;
  if (process.env.WANDB_API_KEY) {
    wandb = new WandBIntegration({
      project: 'gepa-aime',
      name: `aime-optimization-${Date.now()}`,
      config: {
        dataset: 'AIME',
        model: 'gpt-4-turbo-preview',
        populationSize: 8,
        generations: 10
      }
    });
    await wandb.init();
  }
  
  // Configure GEPA optimizer
  const optimizer = new GEPAOptimizer({
    adapter,
    config: {
      populationSize: 8,
      generations: 10,
      mutationRate: 0.3,
      mergeRate: 0.2,
      tournamentSize: 3,
      eliteRatio: 0.2,
      verbose: true,
      runDir: join(process.cwd(), 'runs', `aime-${Date.now()}`)
    },
    wandbIntegration: wandb
  });
  
  // Define initial prompts to optimize
  const initialComponents = {
    math_prompt: `You are an expert mathematician solving competition problems.

When solving problems:
1. Read the problem carefully and identify what is being asked
2. List the given information
3. Choose an appropriate method or formula
4. Show all work step by step
5. Check your answer makes sense
6. Box your final answer

Be precise with calculations and verify each step.`,
    
    format_prompt: `Format your solution as follows:
- Problem Analysis: [What is being asked]
- Given: [List key information]
- Method: [Your approach]
- Solution: [Step-by-step work]
- Answer: [Final boxed answer]`,
    
    verification_prompt: `After solving, verify your answer by:
- Checking it satisfies all conditions
- Confirming units and magnitude make sense
- Trying an alternative method if time permits`
  };
  
  // Run optimization
  console.log('\n🚀 Starting GEPA optimization...');
  console.log('Components to optimize:', Object.keys(initialComponents));
  
  const result = await optimizer.optimize({
    initialCandidate: initialComponents,
    trainData: trainProblems.slice(0, 50), // Use subset for efficiency
    valData: valProblems.slice(0, 20),
    componentsToUpdate: ['math_prompt', 'format_prompt']
  });
  
  // Save results
  const outputDir = join(process.cwd(), 'results', 'aime');
  
  console.log('\n📊 Optimization Results:');
  console.log('Best validation score:', result.bestScore.toFixed(3));
  console.log('Generations completed:', result.generationsCompleted);
  console.log('Total evaluations:', result.totalEvaluations);
  
  // Save optimized prompts
  const outputPath = join(outputDir, 'optimized_prompts.json');
  writeFileSync(outputPath, JSON.stringify({
    metadata: {
      dataset: 'AIME',
      score: result.bestScore,
      generations: result.generationsCompleted,
      timestamp: new Date().toISOString()
    },
    prompts: result.bestCandidate,
    history: result.history
  }, null, 2));
  
  console.log(`\n✅ Results saved to: ${outputPath}`);
  
  // Test on a few examples
  console.log('\n🧪 Testing optimized prompts on sample problems:');
  const testProblems = valProblems.slice(20, 23);
  
  for (const problem of testProblems) {
    console.log('\nProblem:', problem.problem.substring(0, 100) + '...');
    console.log('Expected:', problem.answer);
    
    const evalResult = await adapter.evaluate(
      [problem],
      result.bestCandidate,
      false
    );
    
    console.log('Generated:', evalResult.outputs[0].answer);
    console.log('Correct:', evalResult.scores[0] === 1.0 ? '✓' : '✗');
  }
  
  // Close WandB
  if (wandb) {
    await wandb.finish();
  }
  
  console.log('\n🎉 AIME optimization complete!');
}

// Run the example
main().catch(error => {
  console.error('❌ Error:', error);
  process.exit(1);
});