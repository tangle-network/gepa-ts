import { BaseAdapter, EvaluationBatch } from '../core/adapter.js';
import { ComponentMap, DataInst, ReflectiveDataset, RolloutOutput, Trajectory } from '../types/index.js';

/**
 * Terminal-bench Adapter for optimizing terminal-use agents
 * Integrates with the Terminus agent and terminal-bench benchmark
 */

export interface TerminalBenchExample extends DataInst {
  task: string;
  initialState: {
    cwd: string;
    files?: Record<string, string>;
    env?: Record<string, string>;
  };
  expectedOutcome: {
    files?: Record<string, string>;
    stdout?: string;
    exitCode?: number;
  };
}

export interface TerminalTrace extends Trajectory {
  commands: Array<{
    command: string;
    stdout: string;
    stderr: string;
    exitCode: number;
    timestamp: number;
  }>;
  finalState: {
    cwd: string;
    files: Record<string, string>;
  };
  success: boolean;
}

export interface TerminalOutput extends RolloutOutput {
  finalState: any;
  success: boolean;
  commandCount: number;
  trace?: TerminalTrace;
}

export interface TerminalBenchAdapterConfig {
  agentExecutor: (
    instruction: string,
    task: string,
    initialState: any
  ) => Promise<TerminalTrace> | TerminalTrace;
  maxCommands?: number;
  timeout?: number;
  verbose?: boolean;
}

export class TerminalBenchAdapter extends BaseAdapter<TerminalBenchExample, TerminalTrace, TerminalOutput> {
  private config: TerminalBenchAdapterConfig;
  
  constructor(config: TerminalBenchAdapterConfig) {
    super();
    this.config = {
      maxCommands: 20,
      timeout: 30000,
      verbose: false,
      ...config
    };
  }
  
  private buildInstruction(template: string, task: string): string {
    // Combine the template with the specific task
    return template
      .replace('{task}', task)
      .replace('{max_commands}', String(this.config.maxCommands))
      .replace('{timeout}', String(this.config.timeout / 1000));
  }
  
  private evaluateOutcome(
    trace: TerminalTrace,
    expected: TerminalBenchExample['expectedOutcome']
  ): number {
    let score = 0;
    let totalChecks = 0;
    
    // Check file creation/modification
    if (expected.files) {
      for (const [filename, expectedContent] of Object.entries(expected.files)) {
        totalChecks++;
        const actualContent = trace.finalState.files[filename];
        
        if (actualContent === expectedContent) {
          score += 1;
        } else if (actualContent && actualContent.includes(expectedContent)) {
          score += 0.5; // Partial credit
        }
      }
    }
    
    // Check command success
    if (trace.success) {
      score += 0.5;
      totalChecks += 0.5;
    }
    
    // Check efficiency (fewer commands is better)
    const efficiencyScore = Math.max(0, 1 - (trace.commands.length / this.config.maxCommands!));
    score += efficiencyScore * 0.5;
    totalChecks += 0.5;
    
    // Check for errors
    const errorCount = trace.commands.filter(cmd => cmd.exitCode !== 0).length;
    const errorPenalty = errorCount / trace.commands.length;
    score -= errorPenalty * 0.5;
    
    return Math.max(0, Math.min(1, score / (totalChecks || 1)));
  }
  
  async evaluate(
    batch: TerminalBenchExample[],
    candidate: ComponentMap,
    captureTraces: boolean = false
  ): Promise<EvaluationBatch<TerminalTrace, TerminalOutput>> {
    const instruction = candidate['terminal_instruction'] || candidate['system_prompt'] || '';
    
    const outputs: TerminalOutput[] = [];
    const scores: number[] = [];
    const trajectories: TerminalTrace[] | null = captureTraces ? [] : null;
    
    for (const example of batch) {
      try {
        // Build full instruction
        const fullInstruction = this.buildInstruction(instruction, example.task);
        
        // Execute agent
        const trace = await this.config.agentExecutor(
          fullInstruction,
          example.task,
          example.initialState
        );
        
        // Evaluate outcome
        const score = this.evaluateOutcome(trace, example.expectedOutcome);
        
        const output: TerminalOutput = {
          finalState: trace.finalState,
          success: trace.success,
          commandCount: trace.commands.length,
          trace: captureTraces ? trace : undefined
        };
        
        outputs.push(output);
        scores.push(score);
        
        if (trajectories) {
          trajectories.push(trace);
        }
        
      } catch (error) {
        // Handle execution failure
        const failedOutput: TerminalOutput = {
          finalState: example.initialState,
          success: false,
          commandCount: 0
        };
        
        outputs.push(failedOutput);
        scores.push(0);
        
        if (trajectories) {
          trajectories.push({
            commands: [],
            finalState: example.initialState as any,
            success: false
          });
        }
        
        if (this.config.verbose) {
          console.error(`Failed to execute task: ${error}`);
        }
      }
    }
    
    return { outputs, scores, trajectories };
  }
  
  async makeReflectiveDataset(
    candidate: ComponentMap,
    evalBatch: EvaluationBatch<TerminalTrace, TerminalOutput>,
    componentsToUpdate: string[]
  ): Promise<ReflectiveDataset> {
    const dataset: ReflectiveDataset = {};
    
    if (!evalBatch.trajectories) {
      return dataset;
    }
    
    for (const componentName of componentsToUpdate) {
      const examples: Array<Record<string, any>> = [];
      
      for (let i = 0; i < evalBatch.trajectories.length; i++) {
        const trace = evalBatch.trajectories[i];
        const score = evalBatch.scores[i];
        
        // Analyze command patterns
        const failedCommands = trace.commands.filter(cmd => cmd.exitCode !== 0);
        const successfulCommands = trace.commands.filter(cmd => cmd.exitCode === 0);
        
        let feedback = '';
        let recommendations: string[] = [];
        
        if (score < 0.3) {
          feedback = 'Task failed. ';
          
          if (failedCommands.length > 0) {
            feedback += `${failedCommands.length} commands failed. `;
            recommendations.push('Check command syntax and arguments');
            recommendations.push('Verify file paths and permissions');
          }
          
          if (trace.commands.length === 0) {
            feedback += 'No commands executed. ';
            recommendations.push('Ensure the agent understands the task');
            recommendations.push('Provide clearer instructions');
          }
          
        } else if (score < 0.7) {
          feedback = 'Task partially completed. ';
          
          if (trace.commands.length > 10) {
            feedback += 'Too many commands used. ';
            recommendations.push('Combine multiple operations');
            recommendations.push('Use more efficient commands');
          }
          
          recommendations.push('Verify all expected outcomes');
          
        } else {
          feedback = 'Task completed successfully. ';
          
          if (trace.commands.length < 5) {
            feedback += 'Efficient solution. ';
          }
        }
        
        // Extract command patterns
        const commandPatterns = trace.commands.map(cmd => {
          const baseCommand = cmd.command.split(' ')[0];
          return baseCommand;
        });
        
        examples.push({
          'Task': evalBatch.trajectories[i] ? 'Terminal task' : 'Unknown',
          'Commands Used': commandPatterns.join(', '),
          'Score': score,
          'Feedback': feedback,
          'Recommendations': recommendations,
          'Failed Commands': failedCommands.map(c => c.command),
          'Successful Commands': successfulCommands.slice(0, 3).map(c => c.command),
          'Current Instruction': candidate[componentName] || ''
        });
      }
      
      dataset[componentName] = examples.slice(0, 5);
    }
    
    return dataset;
  }
  
  proposeNewTexts(
    candidate: ComponentMap,
    reflectiveDataset: ReflectiveDataset,
    componentsToUpdate: string[]
  ): ComponentMap {
    const newTexts: ComponentMap = {};
    
    for (const componentName of componentsToUpdate) {
      const examples = reflectiveDataset[componentName] || [];
      const currentInstruction = candidate[componentName] || '';
      
      // Analyze patterns
      const allRecommendations = examples.flatMap(ex => ex['Recommendations'] || []);
      const failedCommands = examples.flatMap(ex => ex['Failed Commands'] || []);
      const successfulCommands = examples.flatMap(ex => ex['Successful Commands'] || []);
      
      // Build improved instruction
      let improvedInstruction = currentInstruction;
      
      // Add specific guidance based on failures
      if (failedCommands.length > 0) {
        const commonFailures = this.findCommonPatterns(failedCommands);
        if (commonFailures.length > 0 && !improvedInstruction.includes('avoid')) {
          improvedInstruction += '\n\nCommon pitfalls to avoid:\n';
          improvedInstruction += commonFailures.map(f => `- ${f}`).join('\n');
        }
      }
      
      // Add successful patterns
      if (successfulCommands.length > 0) {
        const commonSuccesses = this.findCommonPatterns(successfulCommands);
        if (commonSuccesses.length > 0 && !improvedInstruction.includes('effective commands')) {
          improvedInstruction += '\n\nEffective command patterns:\n';
          improvedInstruction += commonSuccesses.slice(0, 3).map(s => `- ${s}`).join('\n');
        }
      }
      
      // Add recommendations
      const uniqueRecommendations = [...new Set(allRecommendations)];
      if (uniqueRecommendations.length > 0 && !improvedInstruction.includes('Remember')) {
        improvedInstruction += '\n\nRemember to:\n';
        improvedInstruction += uniqueRecommendations.slice(0, 3).map(r => `- ${r}`).join('\n');
      }
      
      // Add terminal-specific guidance if missing
      if (!improvedInstruction.includes('tmux')) {
        improvedInstruction += '\n\nNote: You operate directly in a tmux session. ' +
          'Use tmux keystrokes for navigation. ' +
          'Append newline to execute commands.';
      }
      
      newTexts[componentName] = improvedInstruction;
    }
    
    return newTexts;
  }
  
  private findCommonPatterns(commands: string[]): string[] {
    const patterns: Map<string, number> = new Map();
    
    for (const cmd of commands) {
      const baseCmd = cmd.split(' ')[0];
      patterns.set(baseCmd, (patterns.get(baseCmd) || 0) + 1);
    }
    
    return Array.from(patterns.entries())
      .filter(([_, count]) => count > 1)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([pattern]) => pattern);
  }
}