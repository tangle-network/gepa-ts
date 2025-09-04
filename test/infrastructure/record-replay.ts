/**
 * Record/Replay infrastructure for deterministic LLM testing
 * Allows recording real LLM responses and replaying them in tests
 */

import { createHash } from 'crypto';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'fs';
import { join, dirname } from 'path';

export interface RecordedCall {
  hash: string;
  input: string;
  output: string;
  timestamp: number;
  metadata?: {
    model?: string;
    temperature?: number;
    maxTokens?: number;
    [key: string]: any;
  };
}

export interface RecordingSession {
  id: string;
  description: string;
  startTime: number;
  endTime?: number;
  calls: RecordedCall[];
  stats: {
    totalCalls: number;
    uniqueInputs: number;
    totalTokensUsed?: number;
    totalCost?: number;
  };
}

export class RecordReplayManager {
  private mode: 'record' | 'replay' | 'passthrough';
  private recordings: Map<string, RecordedCall> = new Map();
  private currentSession?: RecordingSession;
  private recordingPath: string;
  private strict: boolean;
  
  constructor(options: {
    mode?: 'record' | 'replay' | 'passthrough';
    recordingPath?: string;
    strict?: boolean;
  } = {}) {
    this.mode = options.mode || 
      (process.env.GEPA_TEST_MODE as any) || 
      'replay';
    this.recordingPath = options.recordingPath || 
      join(process.cwd(), 'test', 'recordings');
    this.strict = options.strict ?? true;
    
    this.ensureRecordingDir();
    
    if (this.mode === 'replay') {
      this.loadRecordings();
    }
  }
  
  private ensureRecordingDir(): void {
    if (!existsSync(this.recordingPath)) {
      mkdirSync(this.recordingPath, { recursive: true });
    }
  }
  
  private hashInput(input: string, metadata?: any): string {
    const data = JSON.stringify({ input, metadata });
    return createHash('sha256').update(data).digest('hex');
  }
  
  private loadRecordings(): void {
    const sessionFiles = [
      'core-optimization.json',
      'adapter-tests.json',
      'proposer-tests.json',
      'integration-tests.json'
    ];
    
    for (const file of sessionFiles) {
      const filePath = join(this.recordingPath, file);
      if (existsSync(filePath)) {
        try {
          const session: RecordingSession = JSON.parse(
            readFileSync(filePath, 'utf-8')
          );
          
          for (const call of session.calls) {
            this.recordings.set(call.hash, call);
          }
          
          console.log(`Loaded ${session.calls.length} recordings from ${file}`);
        } catch (error) {
          console.warn(`Failed to load recordings from ${file}:`, error);
        }
      }
    }
    
    console.log(`Total recordings loaded: ${this.recordings.size}`);
  }
  
  private saveSession(): void {
    if (!this.currentSession) return;
    
    this.currentSession.endTime = Date.now();
    
    const filename = `${this.currentSession.id}.json`;
    const filePath = join(this.recordingPath, filename);
    
    writeFileSync(
      filePath,
      JSON.stringify(this.currentSession, null, 2)
    );
    
    console.log(`Saved recording session to ${filename}`);
    console.log(`Total calls recorded: ${this.currentSession.calls.length}`);
  }
  
  startSession(id: string, description: string): void {
    if (this.mode !== 'record') {
      throw new Error('Can only start session in record mode');
    }
    
    this.currentSession = {
      id,
      description,
      startTime: Date.now(),
      calls: [],
      stats: {
        totalCalls: 0,
        uniqueInputs: 0
      }
    };
  }
  
  endSession(): void {
    if (this.mode === 'record' && this.currentSession) {
      this.saveSession();
      this.currentSession = undefined;
    }
  }
  
  async execute(
    input: string,
    actualExecutor: (input: string) => Promise<string>,
    metadata?: any
  ): Promise<string> {
    const hash = this.hashInput(input, metadata);
    
    switch (this.mode) {
      case 'record':
        // Execute and record
        const output = await actualExecutor(input);
        
        const recordedCall: RecordedCall = {
          hash,
          input,
          output,
          timestamp: Date.now(),
          metadata
        };
        
        this.recordings.set(hash, recordedCall);
        
        if (this.currentSession) {
          this.currentSession.calls.push(recordedCall);
          this.currentSession.stats.totalCalls++;
        }
        
        return output;
        
      case 'replay':
        // Return recorded response
        const recorded = this.recordings.get(hash);
        
        if (!recorded) {
          if (this.strict) {
            throw new Error(
              `No recording found for input hash: ${hash}\n` +
              `Input preview: ${input.substring(0, 100)}...`
            );
          } else {
            console.warn(`No recording found for input, using mock response`);
            return this.generateMockResponse(input);
          }
        }
        
        return recorded.output;
        
      case 'passthrough':
        // Execute without recording
        return await actualExecutor(input);
        
      default:
        throw new Error(`Invalid mode: ${this.mode}`);
    }
  }
  
  private generateMockResponse(input: string): string {
    // Generate deterministic mock response based on input
    if (input.includes('math') || input.includes('calculate')) {
      return 'The answer is 42.';
    }
    if (input.includes('sentiment')) {
      return 'positive';
    }
    if (input.includes('classify')) {
      return 'category_a';
    }
    if (input.includes('generate')) {
      return 'Generated text based on the input.';
    }
    return 'Mock response for testing.';
  }
  
  getStats(): {
    mode: string;
    recordingsLoaded: number;
    currentSessionCalls?: number;
  } {
    return {
      mode: this.mode,
      recordingsLoaded: this.recordings.size,
      currentSessionCalls: this.currentSession?.calls.length
    };
  }
  
  /**
   * Create a wrapped LLM function that uses record/replay
   */
  wrapLLM(
    actualLLM?: (prompt: string) => Promise<string>
  ): (prompt: string) => Promise<string> {
    return async (prompt: string) => {
      return this.execute(
        prompt,
        actualLLM || (async () => 'Mock LLM response'),
        { timestamp: Date.now() }
      );
    };
  }
}

/**
 * Test helper for setting up record/replay in tests
 */
export class TestLLM {
  private manager: RecordReplayManager;
  private callCount: number = 0;
  
  constructor(options: {
    sessionId?: string;
    mode?: 'record' | 'replay' | 'passthrough';
    realLLM?: (prompt: string) => Promise<string>;
  } = {}) {
    this.manager = new RecordReplayManager({
      mode: options.mode || 'replay'
    });
    
    if (options.mode === 'record' && options.sessionId) {
      this.manager.startSession(
        options.sessionId,
        `Test recording session: ${options.sessionId}`
      );
    }
  }
  
  async generate(prompt: string): Promise<string> {
    this.callCount++;
    
    return this.manager.execute(
      prompt,
      async () => {
        // Default mock implementation
        if (prompt.includes('improve')) {
          return 'Here is an improved version of the text.';
        }
        if (prompt.includes('merge')) {
          return 'Merged text combining both inputs.';
        }
        if (prompt.includes('evaluate')) {
          return '0.85';
        }
        return 'Generated response.';
      },
      { callNumber: this.callCount }
    );
  }
  
  getCallCount(): number {
    return this.callCount;
  }
  
  finish(): void {
    this.manager.endSession();
  }
}

/**
 * Decorator for test functions to enable record/replay
 */
export function withRecordReplay(
  mode: 'record' | 'replay' = 'replay',
  sessionId?: string
) {
  return function(
    target: any,
    propertyKey: string,
    descriptor: PropertyDescriptor
  ) {
    const originalMethod = descriptor.value;
    
    descriptor.value = async function(...args: any[]) {
      const testLLM = new TestLLM({ mode, sessionId });
      
      try {
        // Inject testLLM as first argument
        const result = await originalMethod.call(this, testLLM, ...args);
        return result;
      } finally {
        testLLM.finish();
      }
    };
    
    return descriptor;
  };
}

/**
 * Utility to record real LLM calls for test data
 */
export async function recordRealLLMCalls(
  sessionId: string,
  calls: Array<{
    input: string;
    execute: () => Promise<string>;
  }>
): Promise<void> {
  const manager = new RecordReplayManager({ mode: 'record' });
  manager.startSession(sessionId, `Recording real LLM calls for ${sessionId}`);
  
  console.log(`Recording ${calls.length} LLM calls...`);
  
  for (let i = 0; i < calls.length; i++) {
    const call = calls[i];
    console.log(`Recording call ${i + 1}/${calls.length}...`);
    
    try {
      await manager.execute(
        call.input,
        call.execute,
        { index: i }
      );
    } catch (error) {
      console.error(`Failed to record call ${i}:`, error);
    }
  }
  
  manager.endSession();
  console.log('Recording complete!');
}

/**
 * Create deterministic test data
 */
export function createTestRecordings(): void {
  const recordings: RecordingSession = {
    id: 'test-recordings',
    description: 'Pre-recorded test data for unit tests',
    startTime: Date.now(),
    endTime: Date.now() + 1000,
    calls: [
      {
        hash: 'test-1',
        input: 'Improve this text: Hello world',
        output: 'Greetings, world!',
        timestamp: Date.now()
      },
      {
        hash: 'test-2',
        input: 'Merge these texts: A and B',
        output: 'A combined with B',
        timestamp: Date.now()
      }
    ],
    stats: {
      totalCalls: 2,
      uniqueInputs: 2
    }
  };
  
  const recordingPath = join(process.cwd(), 'test', 'recordings');
  if (!existsSync(recordingPath)) {
    mkdirSync(recordingPath, { recursive: true });
  }
  
  writeFileSync(
    join(recordingPath, 'test-recordings.json'),
    JSON.stringify(recordings, null, 2)
  );
  
  console.log('Created test recordings');
}