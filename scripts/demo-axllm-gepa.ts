#!/usr/bin/env npx tsx

/**
 * Interactive Demo: AxLLM + GEPA with Real AI
 * 
 * This script demonstrates GEPA evolving AxLLM programs with actual AI calls.
 * It's interactive and shows the evolution process step by step.
 * 
 * Usage:
 *   export OPENAI_API_KEY=sk-...
 *   npx tsx scripts/demo-axllm-gepa.ts
 * 
 * Or with npm script:
 *   npm run demo:axllm
 */

import * as readline from 'readline/promises';
import { stdin, stdout } from 'process';
import OpenAI from 'openai';
import { GEPAOptimizer } from '../src/core/optimizer.js';
import { createAxLLMNativeAdapter } from '../src/adapters/axllm-native-adapter.js';

// Colors for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

// Minimal AxLLM implementation for demo
class DemoAxProgram {
  signature: string;
  demos: any[] = [];
  modelConfig: any = { temperature: 0.7, model: 'gpt-3.5-turbo' };
  
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
      for (const demo of this.demos) {
        const input = demo.input || demo.text || demo.question || JSON.stringify(demo);
        const output = demo.output || demo.label || demo.answer || demo.sentiment;
        prompt += `Input: ${input}\nOutput: ${output}\n\n`;
      }
    }
    
    const inputStr = inputs.input || inputs.text || inputs.question || JSON.stringify(inputs);
    prompt += `Input: ${inputStr}\nOutput: `;
    
    return prompt;
  }
  
  private parseResponse(response: string): any {
    if (this.signature.toLowerCase().includes('sentiment')) {
      const lower = response.toLowerCase();
      if (lower.includes('positive')) return { sentiment: 'positive' };
      if (lower.includes('negative')) return { sentiment: 'negative' };
      if (lower.includes('neutral')) return { sentiment: 'neutral' };
    }
    
    if (this.signature.toLowerCase().includes('yes') || this.signature.toLowerCase().includes('no')) {
      const lower = response.toLowerCase();
      if (lower.includes('yes')) return { answer: 'yes' };
      if (lower.includes('no')) return { answer: 'no' };
    }
    
    return { output: response.trim() };
  }
}

async function main() {
  const rl = readline.createInterface({ input: stdin, output: stdout });
  
  console.log(`\n${colors.bright}${colors.cyan}╔════════════════════════════════════════════╗${colors.reset}`);
  console.log(`${colors.bright}${colors.cyan}║     GEPA + AxLLM Interactive Demo         ║${colors.reset}`);
  console.log(`${colors.bright}${colors.cyan}║     Real AI Evolution with OpenAI         ║${colors.reset}`);
  console.log(`${colors.bright}${colors.cyan}╚════════════════════════════════════════════╝${colors.reset}\n`);
  
  // Check API key
  if (!process.env.OPENAI_API_KEY) {
    console.log(`${colors.red}❌ OpenAI API key not found!${colors.reset}\n`);
    console.log('Please set your API key:');
    console.log(`  ${colors.dim}export OPENAI_API_KEY=sk-...${colors.reset}\n`);
    
    const answer = await rl.question('Enter API key now (or press Enter to exit): ');
    if (!answer) {
      rl.close();
      return;
    }
    process.env.OPENAI_API_KEY = answer;
  }
  
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  
  // LLM wrapper
  const llm = {
    async generate(prompt: string, config: any = {}) {
      const response = await openai.chat.completions.create({
        model: config.model || 'gpt-3.5-turbo',
        messages: [{ role: 'user', content: prompt }],
        temperature: config.temperature || 0.7,
        max_tokens: config.maxTokens || 50
      });
      return response.choices[0]?.message?.content || '';
    }
  };
  
  // Choose task
  console.log(`${colors.bright}Choose a task to optimize:${colors.reset}`);
  console.log('1. Sentiment Analysis');
  console.log('2. Yes/No Question Answering');
  console.log('3. Text Classification');
  console.log('4. Custom (you define)\n');
  
  const choice = await rl.question('Enter choice (1-4): ');
  
  let task: any = {};
  
  switch (choice) {
    case '1':
      task = {
        name: 'Sentiment Analysis',
        initialSignature: 'Analyze sentiment',
        data: [
          { text: "This is amazing! Best purchase ever!", sentiment: "positive" },
          { text: "Terrible quality, very disappointed", sentiment: "negative" },
          { text: "It works fine, nothing special", sentiment: "neutral" },
          { text: "Absolutely fantastic product!", sentiment: "positive" },
          { text: "Complete waste of money", sentiment: "negative" },
          { text: "Average quality for the price", sentiment: "neutral" }
        ],
        metric: (pred: any, example: any) => 
          pred.sentiment === example.sentiment ? 1 : 0,
        testInput: { text: "The product is okay but overpriced" }
      };
      break;
      
    case '2':
      task = {
        name: 'Yes/No Questions',
        initialSignature: 'Answer with yes or no',
        data: [
          { question: "Is the sky blue?", answer: "yes" },
          { question: "Can fish fly?", answer: "no" },
          { question: "Is 2+2 equal to 4?", answer: "yes" },
          { question: "Is Paris in Germany?", answer: "no" },
          { question: "Do humans need water?", answer: "yes" }
        ],
        metric: (pred: any, example: any) => 
          pred.answer === example.answer ? 1 : 0,
        testInput: { question: "Is gold heavier than silver?" }
      };
      break;
      
    case '3':
      task = {
        name: 'Text Classification',
        initialSignature: 'Classify text category',
        data: [
          { text: "The stock market rose 5% today", label: "finance" },
          { text: "New smartphone released with AI features", label: "technology" },
          { text: "Local team wins championship", label: "sports" },
          { text: "Bitcoin hits new record high", label: "finance" },
          { text: "Scientists discover new planet", label: "science" }
        ],
        metric: (pred: any, example: any) => 
          pred.label === example.label ? 1 : 0,
        testInput: { text: "Apple announces new MacBook Pro" }
      };
      break;
      
    default:
      console.log('\n📝 Define your custom task:\n');
      const signature = await rl.question('Enter task signature: ');
      const exampleInput = await rl.question('Example input field name: ');
      const exampleOutput = await rl.question('Example output field name: ');
      
      task = {
        name: 'Custom Task',
        initialSignature: signature,
        data: [
          { [exampleInput]: "Example 1", [exampleOutput]: "Output 1" },
          { [exampleInput]: "Example 2", [exampleOutput]: "Output 2" }
        ],
        metric: (pred: any, example: any) => 
          pred[exampleOutput] === example[exampleOutput] ? 1 : 0,
        testInput: { [exampleInput]: "Test input" }
      };
  }
  
  console.log(`\n${colors.bright}${colors.green}✅ Task: ${task.name}${colors.reset}\n`);
  
  // Configure optimization
  console.log(`${colors.bright}Configure GEPA optimization:${colors.reset}`);
  const populationStr = await rl.question('Population size (default 3): ') || '3';
  const generationsStr = await rl.question('Generations (default 2): ') || '2';
  
  const populationSize = parseInt(populationStr);
  const generations = parseInt(generationsStr);
  
  console.log(`\n${colors.yellow}⚠️  This will make ~${populationSize * generations * task.data.length} API calls${colors.reset}`);
  const confirm = await rl.question('Continue? (y/n): ');
  
  if (confirm.toLowerCase() !== 'y') {
    console.log('Cancelled.');
    rl.close();
    return;
  }
  
  // Create adapter
  const adapter = createAxLLMNativeAdapter({
    axInstance: (sig: string) => new DemoAxProgram(sig),
    llm,
    baseProgram: {
      signature: task.initialSignature,
      demos: [],
      modelConfig: { temperature: 0.7 }
    },
    metricFn: ({ prediction, example }) => task.metric(prediction, example),
    reflectionLm: async (prompt) => {
      // Use AI to evolve
      console.log(`  ${colors.dim}🤔 Asking AI to evolve program...${colors.reset}`);
      
      const response = await llm.generate(
        prompt + '\n\nReturn only valid JSON.',
        { temperature: 0.8, maxTokens: 300 }
      );
      
      const match = response.match(/\{[\s\S]*\}/);
      if (match) return match[0];
      
      // Fallback evolution
      return JSON.stringify({
        signature: `${task.initialSignature} - be precise and accurate`,
        demos: task.data.slice(0, 3),
        modelConfig: { temperature: 0.3 }
      });
    }
  });
  
  // Run optimization
  console.log(`\n${colors.bright}${colors.blue}🚀 Starting GEPA Evolution...${colors.reset}\n`);
  
  const optimizer = new GEPAOptimizer({
    adapter,
    config: {
      populationSize,
      generations,
      mutationRate: 0.5,
      verbose: false // We'll do custom logging
    }
  });
  
  // Track progress
  let currentGen = 0;
  const startTime = Date.now();
  
  // Custom logging
  const originalLog = console.log;
  console.log = (...args) => {
    const msg = args.join(' ');
    if (msg.includes('Generation')) {
      currentGen++;
      originalLog(`\n${colors.bright}Generation ${currentGen}/${generations}${colors.reset}`);
    } else if (msg.includes('Score:')) {
      originalLog(`  ${colors.green}${msg}${colors.reset}`);
    }
  };
  
  try {
    const result = await optimizer.optimize({
      initialCandidate: { 
        program: JSON.stringify({
          signature: task.initialSignature,
          demos: []
        })
      },
      trainData: task.data.slice(0, Math.ceil(task.data.length * 0.7)),
      valData: task.data.slice(Math.ceil(task.data.length * 0.7)),
      componentsToUpdate: ['program'],
      maxMetricCalls: populationSize * generations * 3
    });
    
    console.log = originalLog; // Restore
    
    const timeElapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    
    // Results
    console.log(`\n${colors.bright}${colors.green}═══════════════════════════════════════${colors.reset}`);
    console.log(`${colors.bright}${colors.green}✅ Optimization Complete!${colors.reset}`);
    console.log(`${colors.bright}${colors.green}═══════════════════════════════════════${colors.reset}\n`);
    
    const finalProgram = JSON.parse(result.bestCandidate.program);
    
    console.log(`📊 ${colors.bright}Results:${colors.reset}`);
    console.log(`  Score: ${colors.green}${(result.bestScore * 100).toFixed(1)}%${colors.reset}`);
    console.log(`  Time: ${timeElapsed}s`);
    console.log(`  Generations: ${result.generation}`);
    console.log(`  Evaluations: ${result.evaluationCount}`);
    
    console.log(`\n📝 ${colors.bright}Evolution Summary:${colors.reset}`);
    console.log(`  ${colors.dim}Initial:${colors.reset}`);
    console.log(`    Signature: "${task.initialSignature}"`);
    console.log(`    Demos: 0`);
    console.log(`    Temperature: 0.7`);
    
    console.log(`\n  ${colors.bright}Final:${colors.reset}`);
    console.log(`    Signature: "${finalProgram.signature}"`);
    console.log(`    Demos: ${finalProgram.demos?.length || 0}`);
    console.log(`    Temperature: ${finalProgram.modelConfig?.temperature || 0.7}`);
    
    // Test evolved program
    console.log(`\n🧪 ${colors.bright}Testing evolved program:${colors.reset}\n`);
    console.log(`  Input: ${JSON.stringify(task.testInput)}`);
    
    const evolvedProgram = new DemoAxProgram(finalProgram.signature);
    if (finalProgram.demos) evolvedProgram.setDemos(finalProgram.demos);
    if (finalProgram.modelConfig) evolvedProgram.modelConfig = finalProgram.modelConfig;
    
    const prediction = await evolvedProgram.forward(llm, task.testInput);
    console.log(`  ${colors.green}Prediction: ${JSON.stringify(prediction)}${colors.reset}`);
    
    // Save option
    console.log(`\n${colors.bright}💾 Save optimization?${colors.reset}`);
    const save = await rl.question('Save to file? (y/n): ');
    
    if (save.toLowerCase() === 'y') {
      const filename = `gepa-${task.name.toLowerCase().replace(/\s+/g, '-')}-${Date.now()}.json`;
      const fs = await import('fs/promises');
      
      await fs.writeFile(
        filename,
        JSON.stringify({
          task: task.name,
          timestamp: new Date().toISOString(),
          optimization: {
            score: result.bestScore,
            generations: result.generation,
            evaluations: result.evaluationCount,
            time: timeElapsed
          },
          program: finalProgram,
          testResult: prediction
        }, null, 2)
      );
      
      console.log(`${colors.green}✅ Saved to ${filename}${colors.reset}`);
    }
    
  } catch (error) {
    console.log = originalLog; // Restore
    console.error(`\n${colors.red}❌ Error: ${error}${colors.reset}`);
  }
  
  rl.close();
  console.log(`\n${colors.bright}Thanks for trying GEPA + AxLLM!${colors.reset}\n`);
}

// Run
main().catch(error => {
  console.error(`${colors.red}Fatal error: ${error}${colors.reset}`);
  process.exit(1);
});