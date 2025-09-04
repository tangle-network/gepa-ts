import { spawn } from 'child_process';
import * as path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export interface DSPyExecutorConfig {
  model?: string;
  apiKey?: string;
  maxTokens?: number;
  temperature?: number;
}

export interface DSPyExample {
  inputs?: Record<string, any>;
  [key: string]: any;
}

export interface DSPyExecutionResult {
  success: boolean;
  results?: Array<{
    outputs: Record<string, any>;
    score: number;
    feedback?: string;
    error?: string;
  }>;
  scores?: number[];
  traces?: any[];
  averageScore?: number;
  error?: string;
  traceback?: string;
}

/**
 * Executes DSPy programs via Python subprocess
 * Provides real execution instead of simulation
 */
export class DSPyExecutor {
  private pythonPath: string;
  private executorPath: string;

  constructor(pythonPath: string = 'python3') {
    this.pythonPath = pythonPath;
    this.executorPath = path.join(__dirname, 'python-dspy-executor.py');
  }

  /**
   * Execute a DSPy program with given examples
   */
  async execute(
    programCode: string,
    examples: DSPyExample[],
    config: DSPyExecutorConfig = {},
    metricFnCode?: string
  ): Promise<DSPyExecutionResult> {
    return new Promise((resolve, reject) => {
      const input = {
        programCode,
        examples,
        taskLmConfig: {
          model: config.model || 'openai/gpt-4o-mini',
          apiKey: config.apiKey || process.env.OPENAI_API_KEY,
          maxTokens: config.maxTokens || 2000,
          temperature: config.temperature || 0.7
        },
        metricFnCode
      };

      const python = spawn(this.pythonPath, [this.executorPath]);
      let stdout = '';
      let stderr = '';

      python.stdout.on('data', (data) => {
        stdout += data.toString();
      });

      python.stderr.on('data', (data) => {
        stderr += data.toString();
      });

      python.on('close', (code) => {
        if (code !== 0) {
          reject(new Error(`Python process exited with code ${code}: ${stderr}`));
          return;
        }

        try {
          const result = JSON.parse(stdout);
          resolve(result);
        } catch (error) {
          reject(new Error(`Failed to parse Python output: ${stdout}\nStderr: ${stderr}`));
        }
      });

      python.on('error', (error) => {
        reject(error);
      });

      // Send input to Python process
      python.stdin.write(JSON.stringify(input));
      python.stdin.end();
    });
  }

  /**
   * Validate that DSPy is installed and accessible
   */
  async validateEnvironment(): Promise<boolean> {
    const testCode = `
import dspy
program = dspy.Predict("question -> answer")
`;

    try {
      const result = await this.execute(testCode, [{ inputs: { question: "test" } }]);
      return result.success;
    } catch (error) {
      console.error('DSPy environment validation failed:', error);
      return false;
    }
  }

  /**
   * Execute with automatic retry on failure
   */
  async executeWithRetry(
    programCode: string,
    examples: DSPyExample[],
    config: DSPyExecutorConfig = {},
    metricFnCode?: string,
    maxRetries: number = 3
  ): Promise<DSPyExecutionResult> {
    let lastError: Error | undefined;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await this.execute(programCode, examples, config, metricFnCode);
      } catch (error) {
        lastError = error as Error;
        if (i < maxRetries - 1) {
          await new Promise(resolve => setTimeout(resolve, 1000 * (i + 1)));
        }
      }
    }
    
    throw lastError || new Error('Max retries exceeded');
  }
}

/**
 * Factory function for creating DSPy executor
 */
export function createDSPyExecutor(pythonPath?: string): DSPyExecutor {
  return new DSPyExecutor(pythonPath);
}