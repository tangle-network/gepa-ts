import * as fs from 'fs';
import * as path from 'path';
import * as crypto from 'crypto';

export interface RecordReplayConfig {
  cacheDir: string;
  record: boolean;
  verbose?: boolean;
}

export class RecordReplayLLM {
  private cache: Map<string, string> = new Map();
  private cacheFile: string;
  private isDirty: boolean = false;
  
  constructor(private config: RecordReplayConfig) {
    this.cacheFile = path.join(config.cacheDir, 'llm_cache.json');
    this.loadCache();
  }
  
  private loadCache(): void {
    if (fs.existsSync(this.cacheFile)) {
      try {
        const data = JSON.parse(fs.readFileSync(this.cacheFile, 'utf-8'));
        this.cache = new Map(Object.entries(data));
        if (this.config.verbose) {
          console.log(`Loaded ${this.cache.size} cached LLM responses`);
        }
      } catch (error) {
        console.warn('Failed to load cache:', error);
        this.cache = new Map();
      }
    }
  }
  
  private saveCache(): void {
    if (!this.isDirty) return;
    
    fs.mkdirSync(this.config.cacheDir, { recursive: true });
    const data = Object.fromEntries(this.cache);
    fs.writeFileSync(this.cacheFile, JSON.stringify(data, null, 2));
    
    if (this.config.verbose) {
      console.log(`Saved ${this.cache.size} LLM responses to cache`);
    }
  }
  
  private getKey(input: string | any): string {
    const content = typeof input === 'string' ? input : JSON.stringify(input);
    return crypto.createHash('sha256').update(content).digest('hex');
  }
  
  /**
   * Create a task LLM function for system evaluation
   */
  createTaskLM(actualLM?: (input: any) => Promise<string> | string): (input: any) => Promise<string> | string {
    return (input: any) => {
      const key = this.getKey(['task', input]);
      
      if (this.config.record) {
        if (this.cache.has(key)) {
          return this.cache.get(key)!;
        }
        
        if (!actualLM) {
          throw new Error('Actual LLM required in record mode');
        }
        
        const result = actualLM(input);
        if (result instanceof Promise) {
          return result.then(res => {
            this.cache.set(key, res);
            this.isDirty = true;
            return res;
          });
        } else {
          this.cache.set(key, result);
          this.isDirty = true;
          return result;
        }
      } else {
        // Replay mode
        if (!this.cache.has(key)) {
          throw new Error(`No cached response for key: ${key.substring(0, 8)}... Run with record=true first.`);
        }
        return this.cache.get(key)!;
      }
    };
  }
  
  /**
   * Create a reflection LLM function for optimization
   */
  createReflectionLM(actualLM?: (prompt: string) => Promise<string> | string): (prompt: string) => Promise<string> | string {
    return (prompt: string) => {
      const key = this.getKey(['reflection', prompt]);
      
      if (this.config.record) {
        if (this.cache.has(key)) {
          return this.cache.get(key)!;
        }
        
        if (!actualLM) {
          throw new Error('Actual LLM required in record mode');
        }
        
        const result = actualLM(prompt);
        if (result instanceof Promise) {
          return result.then(res => {
            this.cache.set(key, res);
            this.isDirty = true;
            return res;
          });
        } else {
          this.cache.set(key, result);
          this.isDirty = true;
          return result;
        }
      } else {
        // Replay mode
        if (!this.cache.has(key)) {
          throw new Error(`No cached response for reflection. Run with record=true first.`);
        }
        return this.cache.get(key)!;
      }
    };
  }
  
  /**
   * Save any pending cache updates
   */
  flush(): void {
    this.saveCache();
  }
  
  /**
   * Get cache statistics
   */
  getStats(): { size: number; hits: number; misses: number } {
    return {
      size: this.cache.size,
      hits: 0, // Would need to track this
      misses: 0 // Would need to track this
    };
  }
}

/**
 * Golden file management for deterministic testing
 */
export class GoldenFileManager {
  constructor(private goldenDir: string) {
    fs.mkdirSync(goldenDir, { recursive: true });
  }
  
  save(name: string, data: any): void {
    const filePath = path.join(this.goldenDir, `${name}.json`);
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
  }
  
  load(name: string): any {
    const filePath = path.join(this.goldenDir, `${name}.json`);
    if (!fs.existsSync(filePath)) {
      throw new Error(`Golden file not found: ${filePath}`);
    }
    return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  }
  
  exists(name: string): boolean {
    const filePath = path.join(this.goldenDir, `${name}.json`);
    return fs.existsSync(filePath);
  }
  
  compare(name: string, actual: any): boolean {
    const expected = this.load(name);
    return JSON.stringify(expected) === JSON.stringify(actual);
  }
}