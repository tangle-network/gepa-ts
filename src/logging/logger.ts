import { LoggerProtocol } from '../types/index.js';

export class StdOutLogger implements LoggerProtocol {
  log(message: string): void {
    console.log(`[GEPA] ${message}`);
  }
}

export class FileLogger implements LoggerProtocol {
  private filePath: string;
  private fs: typeof import('fs');
  
  constructor(filePath: string) {
    this.filePath = filePath;
    this.fs = require('fs');
  }
  
  log(message: string): void {
    const timestamp = new Date().toISOString();
    const logMessage = `[${timestamp}] ${message}\n`;
    this.fs.appendFileSync(this.filePath, logMessage);
  }
}

export class MultiLogger implements LoggerProtocol {
  private loggers: LoggerProtocol[];
  
  constructor(...loggers: LoggerProtocol[]) {
    this.loggers = loggers;
  }
  
  log(message: string): void {
    for (const logger of this.loggers) {
      logger.log(message);
    }
  }
}

export class SilentLogger implements LoggerProtocol {
  log(_message: string): void {
    // Silent logger does nothing
  }
}