#!/usr/bin/env tsx
/**
 * Terminal-bench training example - Terminal agent optimization
 * Demonstrates optimizing instructions for terminal-using agents
 */

import { GEPAOptimizer } from '../src/core/optimizer.js';
import { TerminalBenchAdapter, TerminalBenchExample, TerminalTrace } from '../src/adapters/terminal-bench-adapter.js';
import { OpenAILanguageModel } from '../src/models/openai.js';
import { WandBIntegration } from '../src/integrations/wandb.js';
import { execSync, spawn } from 'child_process';
import { writeFileSync, mkdirSync, rmSync } from 'fs';
import { join, dirname } from 'path';
import { tmpdir } from 'os';

/**
 * Mock terminal executor for testing
 * In production, this would integrate with actual Terminus agent
 */
class MockTerminalExecutor {
  private lm: OpenAILanguageModel;
  
  constructor(lm: OpenAILanguageModel) {
    this.lm = lm;
  }
  
  async execute(
    instruction: string,
    task: string,
    initialState: any
  ): Promise<TerminalTrace> {
    const tmpDir = join(tmpdir(), `terminal-bench-${Date.now()}`);
    mkdirSync(tmpDir, { recursive: true });
    
    // Create initial files
    if (initialState.files) {
      for (const [path, content] of Object.entries(initialState.files)) {
        const fullPath = join(tmpDir, path);
        mkdirSync(dirname(fullPath), { recursive: true });
        writeFileSync(fullPath, content as string);
      }
    }
    
    // Generate commands using LLM
    const prompt = `${instruction}

Task: ${task}

Current directory: ${tmpDir}
Available files: ${Object.keys(initialState.files || {}).join(', ')}

Generate a sequence of bash commands to complete this task.
Output only the commands, one per line, no explanations.`;
    
    const response = await this.lm.generate(prompt);
    const commands = response.split('\n')
      .filter(line => line.trim() && !line.startsWith('#'))
      .slice(0, 10); // Limit commands
    
    // Execute commands and capture trace
    const trace: TerminalTrace = {
      commands: [],
      finalState: {
        cwd: tmpDir,
        files: {}
      },
      success: true
    };
    
    for (const cmd of commands) {
      try {
        const result = execSync(cmd, {
          cwd: tmpDir,
          encoding: 'utf-8',
          stdio: 'pipe'
        });
        
        trace.commands.push({
          command: cmd,
          stdout: result,
          stderr: '',
          exitCode: 0,
          timestamp: Date.now()
        });
      } catch (error: any) {
        trace.commands.push({
          command: cmd,
          stdout: '',
          stderr: error.stderr || error.message,
          exitCode: error.status || 1,
          timestamp: Date.now()
        });
        trace.success = false;
        break;
      }
    }
    
    // Capture final state
    try {
      const files = execSync('find . -type f', {
        cwd: tmpDir,
        encoding: 'utf-8'
      }).split('\n').filter(f => f);
      
      for (const file of files) {
        const content = execSync(`cat ${file}`, {
          cwd: tmpDir,
          encoding: 'utf-8'
        });
        trace.finalState.files[file] = content;
      }
    } catch (error) {
      // Ignore errors reading final state
    }
    
    // Cleanup
    try {
      rmSync(tmpDir, { recursive: true, force: true });
    } catch {
      // Ignore cleanup errors
    }
    
    return trace;
  }
}

// Example terminal-bench tasks
const TERMINAL_BENCH_TASKS: TerminalBenchExample[] = [
  {
    task: 'Create a Python script that calculates fibonacci numbers',
    initialState: {
      cwd: '.',
      files: {}
    },
    expectedOutcome: {
      files: {
        './fibonacci.py': 'def fibonacci'
      }
    }
  },
  {
    task: 'Set up a basic Node.js project with package.json',
    initialState: {
      cwd: '.',
      files: {}
    },
    expectedOutcome: {
      files: {
        './package.json': '"name":'
      }
    }
  },
  {
    task: 'Create a README.md with project title and description',
    initialState: {
      cwd: '.',
      files: {
        'main.py': 'print("Hello World")'
      }
    },
    expectedOutcome: {
      files: {
        './README.md': '# '
      }
    }
  },
  {
    task: 'Write a bash script that counts files in a directory',
    initialState: {
      cwd: '.',
      files: {}
    },
    expectedOutcome: {
      files: {
        './count_files.sh': '#!/bin/bash'
      }
    }
  },
  {
    task: 'Create a simple HTML page with a title and heading',
    initialState: {
      cwd: '.',
      files: {}
    },
    expectedOutcome: {
      files: {
        './index.html': '<html>'
      }
    }
  }
];

async function main() {
  console.log('🖥️  Starting Terminal-bench Agent Optimization');
  
  // Initialize language model
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    console.error('❌ OPENAI_API_KEY environment variable not set');
    process.exit(1);
  }
  
  const lm = new OpenAILanguageModel({
    apiKey,
    model: 'gpt-4-turbo-preview',
    temperature: 0.5,
    maxTokens: 500
  });
  
  // Create mock executor
  const executor = new MockTerminalExecutor(lm);
  
  // Create adapter
  const adapter = new TerminalBenchAdapter({
    agentExecutor: (instruction, task, state) => 
      executor.execute(instruction, task, state),
    maxCommands: 15,
    timeout: 30000,
    verbose: true
  });
  
  // Initialize WandB (optional)
  let wandb: WandBIntegration | undefined;
  if (process.env.WANDB_API_KEY) {
    wandb = new WandBIntegration({
      project: 'gepa-terminal-bench',
      name: `terminal-optimization-${Date.now()}`,
      config: {
        dataset: 'Terminal-bench',
        model: 'gpt-4-turbo-preview',
        populationSize: 6,
        generations: 8
      }
    });
    await wandb.init();
  }
  
  // Configure GEPA optimizer
  const optimizer = new GEPAOptimizer({
    adapter,
    config: {
      populationSize: 6,
      generations: 8,
      mutationRate: 0.4,
      mergeRate: 0.2,
      tournamentSize: 2,
      eliteRatio: 0.25,
      verbose: true,
      runDir: join(process.cwd(), 'runs', `terminal-${Date.now()}`)
    },
    wandbIntegration: wandb
  });
  
  // Define initial instructions to optimize
  const initialComponents = {
    terminal_instruction: `You are a terminal agent that executes bash commands to complete tasks.

Guidelines:
- Read the task carefully
- Plan your approach before executing
- Use standard Unix commands
- Create files and directories as needed
- Verify your work with ls, cat, etc.

Task: {task}

Execute commands one at a time. Be efficient and precise.`,
    
    error_handling: `If a command fails:
- Check the error message
- Try an alternative approach
- Create parent directories if needed
- Use appropriate permissions`,
    
    verification: `After completing the task:
- List created files with ls
- Show file contents with cat
- Confirm task requirements are met`
  };
  
  // Split data
  const trainTasks = TERMINAL_BENCH_TASKS.slice(0, 3);
  const valTasks = TERMINAL_BENCH_TASKS.slice(3);
  
  console.log('\n🚀 Starting GEPA optimization...');
  console.log(`Training tasks: ${trainTasks.length}`);
  console.log(`Validation tasks: ${valTasks.length}`);
  console.log('Components to optimize:', Object.keys(initialComponents));
  
  const result = await optimizer.optimize({
    initialCandidate: initialComponents,
    trainData: trainTasks,
    valData: valTasks,
    componentsToUpdate: ['terminal_instruction']
  });
  
  // Save results
  const outputDir = join(process.cwd(), 'results', 'terminal-bench');
  mkdirSync(outputDir, { recursive: true });
  
  console.log('\n📊 Optimization Results:');
  console.log('Best validation score:', result.bestScore.toFixed(3));
  console.log('Generations completed:', result.generationsCompleted);
  console.log('Total evaluations:', result.totalEvaluations);
  
  // Save optimized instructions
  const outputPath = join(outputDir, 'optimized_instructions.json');
  writeFileSync(outputPath, JSON.stringify({
    metadata: {
      dataset: 'Terminal-bench',
      score: result.bestScore,
      generations: result.generationsCompleted,
      timestamp: new Date().toISOString()
    },
    instructions: result.bestCandidate,
    history: result.history
  }, null, 2));
  
  console.log(`\n✅ Results saved to: ${outputPath}`);
  
  // Show optimized instruction
  console.log('\n📝 Optimized Terminal Instruction:');
  console.log('---');
  console.log(result.bestCandidate.terminal_instruction);
  console.log('---');
  
  // Test on validation task
  console.log('\n🧪 Testing optimized instruction on validation task:');
  const testTask = valTasks[0];
  console.log('Task:', testTask.task);
  
  const evalResult = await adapter.evaluate(
    [testTask],
    result.bestCandidate,
    true
  );
  
  console.log('Success:', evalResult.outputs[0].success ? '✓' : '✗');
  console.log('Commands used:', evalResult.outputs[0].commandCount);
  console.log('Score:', evalResult.scores[0].toFixed(2));
  
  if (evalResult.trajectories && evalResult.trajectories[0]) {
    console.log('\nCommand trace:');
    for (const cmd of evalResult.trajectories[0].commands.slice(0, 5)) {
      console.log(`  $ ${cmd.command}`);
      if (cmd.exitCode !== 0) {
        console.log(`    ❌ Error: ${cmd.stderr}`);
      }
    }
  }
  
  // Close WandB
  if (wandb) {
    await wandb.finish();
  }
  
  console.log('\n🎉 Terminal-bench optimization complete!');
}

// Run the example
main().catch(error => {
  console.error('❌ Error:', error);
  process.exit(1);
});