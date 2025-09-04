import {
  ComponentMap,
  DataInst,
  LoggerProtocol,
  RolloutOutput,
  Trajectory,
  WandBConfig
} from '../types/index.js';
import { GEPAState, initializeGEPAState } from './state.js';
import { ReflectiveMutationProposer } from '../proposer/reflective-mutation.js';
import { MergeProposer } from '../proposer/merge.js';

export interface GEPAEngineConfig<D = DataInst> {
  runDir: string | null;
  evaluator: (inputs: D[], prog: ComponentMap) => Promise<[RolloutOutput[], number[]]> | [RolloutOutput[], number[]];
  valset: D[];
  seedCandidate: ComponentMap;
  maxMetricCalls: number;
  perfectScore: number;
  seed: number;
  reflectiveProposer: ReflectiveMutationProposer<D>;
  mergeProposer?: MergeProposer<D> | null;
  logger: LoggerProtocol;
  useWandB?: boolean;
  wandBConfig?: WandBConfig;
  trackBestOutputs?: boolean;
  displayProgressBar?: boolean;
  raiseOnException?: boolean;
}

export class GEPAEngine<D = DataInst, T = Trajectory, O = RolloutOutput> {
  private config: GEPAEngineConfig<D>;
  
  constructor(config: GEPAEngineConfig<D>) {
    if (!config.maxMetricCalls) {
      throw new Error('maxMetricCalls must be set');
    }
    this.config = {
      ...config,
      runDir: config.runDir || null,
      useWandB: config.useWandB || false,
      trackBestOutputs: config.trackBestOutputs || false,
      displayProgressBar: config.displayProgressBar || false,
      raiseOnException: config.raiseOnException !== false
    };
  }
  
  private valEvaluator() {
    return (prog: ComponentMap) => this.config.evaluator(this.config.valset, prog);
  }
  
  private async runFullEvalAndAdd(
    newProgram: ComponentMap,
    state: GEPAState,
    parentProgramIdx: number[]
  ): Promise<[number, number]> {
    const numMetricCallsByDiscovery = state.totalNumEvals;
    
    const evalResult = this.valEvaluator()(newProgram);
    const [valsetOutputs, valsetSubscores] = evalResult instanceof Promise ? 
      await evalResult : evalResult;
    
    const valsetScore = valsetSubscores.reduce((a, b) => a + b, 0) / valsetSubscores.length;
    
    state.numFullDsEvals += 1;
    state.totalNumEvals += valsetSubscores.length;
    
    const [newProgramIdx, linearParetoFrontProgramIdx] = state.updateStateWithNewProgram(
      parentProgramIdx,
      newProgram,
      valsetScore,
      valsetOutputs,
      valsetSubscores,
      this.config.runDir,
      numMetricCallsByDiscovery
    );
    
    state.fullProgramTrace[state.fullProgramTrace.length - 1].newProgramIdx = newProgramIdx;
    
    if (newProgramIdx === linearParetoFrontProgramIdx) {
      this.config.logger.log(`Iteration ${state.i + 1}: New program is on the linear pareto front`);
    }
    
    this.config.logger.log(
      `Iteration ${state.i + 1}: New program score: ${valsetScore.toFixed(4)} ` +
      `(improvement: ${(valsetScore - state.programFullScoresValSet[0]).toFixed(4)})`
    );
    
    return [newProgramIdx, linearParetoFrontProgramIdx];
  }
  
  async run(): Promise<GEPAState> {
    if (!this.config.valset) {
      throw new Error('valset must be provided to GEPAEngine.run()');
    }
    
    const stateOrPromise = initializeGEPAState(
      this.config.runDir,
      this.config.seedCandidate,
      this.valEvaluator(),
      this.config.trackBestOutputs
    );
    
    const state = stateOrPromise instanceof Promise ? await stateOrPromise : stateOrPromise;
    
    this.config.logger.log(
      `Iteration ${state.i + 1}: Base program full valset score: ${state.programFullScoresValSet[0]}`
    );
    
    if (this.config.mergeProposer) {
      this.config.mergeProposer.lastIterFoundNewProgram = false;
    }
    
    let progressInterval: NodeJS.Timeout | null = null;
    if (this.config.displayProgressBar) {
      progressInterval = setInterval(() => {
        const progress = (state.totalNumEvals / this.config.maxMetricCalls * 100).toFixed(1);
        process.stdout.write(`\rGEPA Progress: ${progress}% [${state.totalNumEvals}/${this.config.maxMetricCalls}]`);
      }, 1000);
    }
    
    try {
      while (state.totalNumEvals < this.config.maxMetricCalls) {
        if (!state.isConsistent()) {
          throw new Error('State consistency check failed');
        }
        
        try {
          state.save(this.config.runDir);
          state.i += 1;
          state.fullProgramTrace.push({ i: state.i });
          
          if (this.config.mergeProposer?.useMerge && 
              this.config.mergeProposer.mergesDue > 0 && 
              this.config.mergeProposer.lastIterFoundNewProgram) {
            
            const proposal = await this.config.mergeProposer.propose(state);
            this.config.mergeProposer.lastIterFoundNewProgram = false;
            
            if (proposal && proposal.tag === 'merge') {
              const parentSums = proposal.subsampleScoresBefore || [Number.NEGATIVE_INFINITY, Number.NEGATIVE_INFINITY];
              const newSum = (proposal.subsampleScoresAfter || []).reduce((a, b) => a + b, 0);
              
              if (newSum >= Math.max(...parentSums)) {
                await this.runFullEvalAndAdd(
                  proposal.candidate,
                  state,
                  proposal.parentProgramIds
                );
                this.config.mergeProposer.mergesDue -= 1;
                this.config.mergeProposer.totalMergesTested += 1;
                continue;
              } else {
                this.config.logger.log(
                  `Iteration ${state.i + 1}: New program subsample score ${newSum} is worse than both parents ${parentSums}, skipping merge`
                );
                // Count evaluations done during merge testing
                if (proposal.subsampleScoresAfter) {
                  state.totalNumEvals += proposal.subsampleScoresAfter.length;
                }
                this.config.mergeProposer.totalMergesTested += 1;
                continue;
              }
            }
          }
          
          if (this.config.mergeProposer) {
            this.config.mergeProposer.lastIterFoundNewProgram = false;
          }
          
          const proposal = await this.config.reflectiveProposer.propose(state);
          if (!proposal) {
            this.config.logger.log(`Iteration ${state.i + 1}: Reflective mutation did not propose a new candidate`);
            // Still increment to avoid infinite loop
            state.totalNumEvals += 1;
            continue;
          }
          
          const oldSum = (proposal.subsampleScoresBefore || []).reduce((a, b) => a + b, 0);
          const newSum = (proposal.subsampleScoresAfter || []).reduce((a, b) => a + b, 0);
          
          if (newSum <= oldSum) {
            this.config.logger.log(`Iteration ${state.i + 1}: New subsample score is not better, skipping`);
            // Still count the evaluations done during subsample testing
            if (proposal.subsampleScoresAfter) {
              state.totalNumEvals += proposal.subsampleScoresAfter.length;
            }
            continue;
          }
          
          await this.runFullEvalAndAdd(
            proposal.candidate,
            state,
            proposal.parentProgramIds
          );
          
          if (this.config.mergeProposer) {
            this.config.mergeProposer.lastIterFoundNewProgram = true;
            if (this.config.mergeProposer.totalMergesTested < this.config.mergeProposer.maxMergeInvocations) {
              this.config.mergeProposer.mergesDue += 1;
            }
          }
          
        } catch (error) {
          this.config.logger.log(`Iteration ${state.i + 1}: Exception during optimization: ${error}`);
          if (error instanceof Error) {
            this.config.logger.log(error.stack || '');
          }
          if (this.config.raiseOnException) {
            throw error;
          } else {
            continue;
          }
        }
      }
    } finally {
      if (progressInterval) {
        clearInterval(progressInterval);
        console.log('');
      }
    }
    
    state.save(this.config.runDir);
    return state;
  }
}