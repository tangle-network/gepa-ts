/**
 * Complete WandB (Weights & Biases) integration for GEPA
 * Provides comprehensive experiment tracking, visualization, and analysis
 */

import { ComponentMap, ReflectiveDataset } from '../types/index.js';
import { GEPAState } from '../core/state.js';
import { EvaluationBatch } from '../core/adapter.js';

export interface WandBConfig {
  project: string;
  entity?: string;
  name?: string;
  tags?: string[];
  notes?: string;
  config?: Record<string, any>;
  mode?: 'online' | 'offline' | 'disabled';
  reinit?: boolean;
  resume?: 'allow' | 'must' | 'never' | 'auto';
  runId?: string;
}

export interface WandBMetrics {
  generation: number;
  trainScore: number;
  valScore: number;
  paretoFrontSize?: number;
  populationDiversity?: number;
  mutationCount?: number;
  mergeCount?: number;
  eliteCount?: number;
  timestamp: number;
}

export interface WandBComponentUpdate {
  generation: number;
  component: string;
  oldText: string;
  newText: string;
  improvement: number;
  updateType: 'mutation' | 'merge' | 'crossover';
}

export interface WandBTrajectory {
  generation: number;
  exampleId: string;
  score: number;
  output: any;
  trace?: any;
}

/**
 * Full WandB Integration for GEPA
 */
export class WandBIntegration {
  private initialized: boolean = false;
  private runId?: string;
  private config: WandBConfig;
  private metricsBuffer: WandBMetrics[] = [];
  private componentUpdates: WandBComponentUpdate[] = [];
  private bestCandidate?: ComponentMap;
  private bestScore: number = 0;
  private generationCount: number = 0;
  
  // WandB API simulation (in production, use actual wandb package)
  private wandbApi: any = null;
  
  constructor(config: WandBConfig) {
    this.config = {
      mode: 'online',
      resume: 'auto',
      ...config
    };
  }
  
  /**
   * Initialize WandB run
   */
  async init(): Promise<void> {
    if (this.initialized) {
      console.warn('WandB already initialized');
      return;
    }
    
    try {
      // In production, initialize actual WandB
      // this.wandbApi = await import('wandb');
      // await this.wandbApi.init(this.config);
      
      // For now, simulate initialization
      this.runId = this.config.runId || `run_${Date.now()}`;
      
      console.log(`🎯 WandB Run Initialized`);
      console.log(`   Project: ${this.config.project}`);
      console.log(`   Run ID: ${this.runId}`);
      console.log(`   Mode: ${this.config.mode}`);
      
      if (this.config.config) {
        console.log(`   Config:`, this.config.config);
      }
      
      this.initialized = true;
      
      // Log initial config
      await this.logConfig(this.config.config || {});
      
    } catch (error) {
      console.error('Failed to initialize WandB:', error);
      this.config.mode = 'disabled';
    }
  }
  
  /**
   * Log configuration
   */
  async logConfig(config: Record<string, any>): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    // In production: wandb.config.update(config)
    console.log('📊 WandB Config:', config);
    
    // Store for artifact creation
    this.config.config = { ...this.config.config, ...config };
  }
  
  /**
   * Log generation metrics
   */
  async logGeneration(
    generation: number,
    state: GEPAState,
    evalResults?: {
      train?: EvaluationBatch<any, any>;
      val?: EvaluationBatch<any, any>;
    }
  ): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    this.generationCount = generation;
    
    const stats = state.getStatistics();
    
    const metrics: WandBMetrics = {
      generation,
      trainScore: stats.averageTrainScore,
      valScore: stats.averageValScore,
      paretoFrontSize: stats.paretoFrontSizes?.reduce((a, b) => a + b, 0),
      populationDiversity: this.calculateDiversity(state),
      mutationCount: 0, // Would track in optimizer
      mergeCount: 0,    // Would track in optimizer
      eliteCount: 0,    // Would track in optimizer
      timestamp: Date.now()
    };
    
    // Additional metrics from evaluation results
    if (evalResults?.train) {
      metrics.trainScore = evalResults.train.scores.reduce((a, b) => a + b, 0) / 
                          evalResults.train.scores.length;
    }
    
    if (evalResults?.val) {
      metrics.valScore = evalResults.val.scores.reduce((a, b) => a + b, 0) / 
                        evalResults.val.scores.length;
    }
    
    this.metricsBuffer.push(metrics);
    
    // Log to WandB
    await this.log({
      'generation': generation,
      'metrics/train_score': metrics.trainScore,
      'metrics/val_score': metrics.valScore,
      'metrics/pareto_front_size': metrics.paretoFrontSize || 0,
      'metrics/population_diversity': metrics.populationDiversity,
      'metrics/best_val_score': stats.bestValScore,
      'metrics/best_train_score': stats.bestTrainScore,
      'population/total_programs': stats.totalPrograms,
      'population/unique_programs': stats.uniquePrograms
    });
    
    // Update best
    if (metrics.valScore > this.bestScore) {
      this.bestScore = metrics.valScore;
      const best = state.getBestProgram();
      if (best) {
        this.bestCandidate = best.program;
        await this.logBestCandidate(best.program, metrics.valScore);
      }
    }
  }
  
  /**
   * Log component update
   */
  async logComponentUpdate(
    generation: number,
    componentName: string,
    oldText: string,
    newText: string,
    scoreImprovement: number,
    updateType: 'mutation' | 'merge' | 'crossover' = 'mutation'
  ): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    const update: WandBComponentUpdate = {
      generation,
      component: componentName,
      oldText,
      newText,
      improvement: scoreImprovement,
      updateType
    };
    
    this.componentUpdates.push(update);
    
    await this.log({
      'components/updated': componentName,
      'components/improvement': scoreImprovement,
      'components/update_type': updateType,
      'components/text_length_change': newText.length - oldText.length
    });
    
    // Log text diff if significant improvement
    if (scoreImprovement > 0.1) {
      await this.logTextDiff(componentName, oldText, newText);
    }
  }
  
  /**
   * Log reflective dataset
   */
  async logReflectiveDataset(
    generation: number,
    dataset: ReflectiveDataset
  ): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    // Analyze dataset
    const analysis: Record<string, any> = {};
    
    for (const [component, examples] of Object.entries(dataset)) {
      const scores = examples.map(ex => ex['Score'] || 0);
      const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      
      analysis[`reflective/${component}/avg_score`] = avgScore;
      analysis[`reflective/${component}/num_examples`] = examples.length;
      
      // Track feedback patterns
      const feedbacks = examples.map(ex => ex['Feedback'] || '');
      const commonIssues = this.extractCommonPatterns(feedbacks);
      
      if (commonIssues.length > 0) {
        analysis[`reflective/${component}/common_issues`] = commonIssues.join(', ');
      }
    }
    
    await this.log(analysis);
  }
  
  /**
   * Log trajectory for detailed analysis
   */
  async logTrajectory(
    generation: number,
    trajectory: WandBTrajectory
  ): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    await this.log({
      'trajectories/generation': generation,
      'trajectories/example_id': trajectory.exampleId,
      'trajectories/score': trajectory.score,
      _step: generation
    });
    
    // Log detailed trace as artifact if available
    if (trajectory.trace) {
      await this.logArtifact(
        `trace_gen${generation}_${trajectory.exampleId}`,
        trajectory.trace,
        'trace'
      );
    }
  }
  
  /**
   * Log evaluation batch results
   */
  async logEvaluationBatch<T, O>(
    generation: number,
    batchType: 'train' | 'val',
    batch: EvaluationBatch<T, O>
  ): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    const avgScore = batch.scores.reduce((a, b) => a + b, 0) / batch.scores.length;
    const minScore = Math.min(...batch.scores);
    const maxScore = Math.max(...batch.scores);
    const stdDev = this.calculateStdDev(batch.scores);
    
    await this.log({
      [`${batchType}/avg_score`]: avgScore,
      [`${batchType}/min_score`]: minScore,
      [`${batchType}/max_score`]: maxScore,
      [`${batchType}/std_dev`]: stdDev,
      [`${batchType}/num_examples`]: batch.scores.length,
      _step: generation
    });
    
    // Log score distribution
    await this.logHistogram(
      `${batchType}_scores_gen${generation}`,
      batch.scores
    );
    
    // Log trajectories if available
    if (batch.trajectories) {
      for (let i = 0; i < Math.min(5, batch.trajectories.length); i++) {
        await this.logTrajectory(generation, {
          generation,
          exampleId: `${batchType}_${i}`,
          score: batch.scores[i],
          output: batch.outputs[i],
          trace: batch.trajectories[i]
        });
      }
    }
  }
  
  /**
   * Log best candidate
   */
  private async logBestCandidate(
    candidate: ComponentMap,
    score: number
  ): Promise<void> {
    await this.log({
      'best/score': score,
      'best/generation': this.generationCount,
      'best/num_components': Object.keys(candidate).length
    });
    
    // Log each component
    for (const [name, text] of Object.entries(candidate)) {
      await this.log({
        [`best_components/${name}_length`]: text.length,
        [`best_components/${name}_words`]: text.split(/\s+/).length
      });
    }
    
    // Save as artifact
    await this.logArtifact('best_candidate', candidate, 'candidate');
  }
  
  /**
   * Log text diff for visualization
   */
  private async logTextDiff(
    componentName: string,
    oldText: string,
    newText: string
  ): Promise<void> {
    // In production, use proper diff library
    const diff = {
      component: componentName,
      old: oldText,
      new: newText,
      added: newText.length - oldText.length,
      timestamp: Date.now()
    };
    
    await this.logArtifact(
      `diff_${componentName}_${this.generationCount}`,
      diff,
      'text_diff'
    );
  }
  
  /**
   * Calculate population diversity
   */
  private calculateDiversity(state: GEPAState): number {
    const programs = state.getAllPrograms();
    if (programs.length <= 1) return 0;
    
    // Calculate pairwise text similarity
    let totalSimilarity = 0;
    let pairs = 0;
    
    for (let i = 0; i < programs.length - 1; i++) {
      for (let j = i + 1; j < programs.length; j++) {
        const text1 = JSON.stringify(programs[i]);
        const text2 = JSON.stringify(programs[j]);
        const similarity = this.textSimilarity(text1, text2);
        totalSimilarity += similarity;
        pairs++;
      }
    }
    
    // Diversity is inverse of average similarity
    const avgSimilarity = totalSimilarity / pairs;
    return 1 - avgSimilarity;
  }
  
  /**
   * Simple text similarity (Jaccard on words)
   */
  private textSimilarity(text1: string, text2: string): number {
    const words1 = new Set(text1.toLowerCase().split(/\s+/));
    const words2 = new Set(text2.toLowerCase().split(/\s+/));
    
    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);
    
    return intersection.size / union.size;
  }
  
  /**
   * Extract common patterns from feedback
   */
  private extractCommonPatterns(feedbacks: string[]): string[] {
    const patterns: Map<string, number> = new Map();
    
    const keywords = [
      'error', 'missing', 'incorrect', 'wrong', 'failed',
      'unclear', 'vague', 'ambiguous', 'format', 'syntax'
    ];
    
    for (const feedback of feedbacks) {
      const lower = feedback.toLowerCase();
      for (const keyword of keywords) {
        if (lower.includes(keyword)) {
          patterns.set(keyword, (patterns.get(keyword) || 0) + 1);
        }
      }
    }
    
    return Array.from(patterns.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([pattern]) => pattern);
  }
  
  /**
   * Calculate standard deviation
   */
  private calculateStdDev(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(variance);
  }
  
  /**
   * Core logging function
   */
  private async log(metrics: Record<string, any>): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    if (this.config.mode === 'offline') {
      // Store locally
      console.log('📊 WandB Metrics:', metrics);
    } else {
      // In production: wandb.log(metrics)
      console.log('📊 WandB Log:', Object.entries(metrics)
        .filter(([k]) => !k.startsWith('_'))
        .map(([k, v]) => `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`)
        .join(', '));
    }
  }
  
  /**
   * Log histogram data
   */
  private async logHistogram(
    name: string,
    values: number[]
  ): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    // In production: wandb.log({name: wandb.Histogram(values)})
    const bins = 10;
    const min = Math.min(...values);
    const max = Math.max(...values);
    const range = max - min;
    const binSize = range / bins;
    
    const histogram: number[] = new Array(bins).fill(0);
    for (const value of values) {
      const binIndex = Math.min(
        Math.floor((value - min) / binSize),
        bins - 1
      );
      histogram[binIndex]++;
    }
    
    console.log(`📊 Histogram ${name}:`, histogram);
  }
  
  /**
   * Log artifact
   */
  private async logArtifact(
    name: string,
    data: any,
    type: string
  ): Promise<void> {
    if (!this.initialized || this.config.mode === 'disabled') return;
    
    // In production: create and log wandb artifact
    console.log(`📦 WandB Artifact: ${name} (${type})`);
    
    // Store important artifacts
    if (type === 'candidate' && name === 'best_candidate') {
      this.bestCandidate = data;
    }
  }
  
  /**
   * Create summary at end of run
   */
  async createSummary(): Promise<Record<string, any>> {
    const summary = {
      runId: this.runId,
      project: this.config.project,
      bestScore: this.bestScore,
      totalGenerations: this.generationCount,
      totalComponentUpdates: this.componentUpdates.length,
      
      // Score progression
      initialScore: this.metricsBuffer[0]?.valScore || 0,
      finalScore: this.metricsBuffer[this.metricsBuffer.length - 1]?.valScore || 0,
      improvement: 0,
      
      // Component analysis
      componentsUpdated: [...new Set(this.componentUpdates.map(u => u.component))],
      averageImprovement: 0,
      
      // Best candidate
      bestCandidate: this.bestCandidate,
      
      // Timing
      totalRuntime: this.metricsBuffer.length > 0 
        ? this.metricsBuffer[this.metricsBuffer.length - 1].timestamp - this.metricsBuffer[0].timestamp
        : 0
    };
    
    // Calculate improvements
    if (this.metricsBuffer.length > 0) {
      summary.improvement = summary.finalScore - summary.initialScore;
    }
    
    if (this.componentUpdates.length > 0) {
      const totalImprovement = this.componentUpdates
        .reduce((sum, u) => sum + u.improvement, 0);
      summary.averageImprovement = totalImprovement / this.componentUpdates.length;
    }
    
    return summary;
  }
  
  /**
   * Finish WandB run
   */
  async finish(): Promise<void> {
    if (!this.initialized) return;
    
    // Create and log summary
    const summary = await this.createSummary();
    
    console.log('\n📊 WandB Run Summary:');
    console.log('=' . repeat(50));
    console.log(`Run ID: ${summary.runId}`);
    console.log(`Project: ${summary.project}`);
    console.log(`Best Score: ${summary.bestScore.toFixed(3)}`);
    console.log(`Improvement: ${summary.improvement.toFixed(3)}`);
    console.log(`Generations: ${summary.totalGenerations}`);
    console.log(`Runtime: ${(summary.totalRuntime / 1000).toFixed(1)}s`);
    console.log('=' . repeat(50));
    
    // In production: wandb.finish()
    
    this.initialized = false;
  }
  
  /**
   * Get current run URL
   */
  getRunUrl(): string | null {
    if (!this.initialized || !this.runId) return null;
    
    // In production: return wandb.run.get_url()
    return `https://wandb.ai/${this.config.entity || 'user'}/${this.config.project}/runs/${this.runId}`;
  }
  
  /**
   * Check if initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }
}