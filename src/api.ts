import { 
  ComponentMap, 
  DataInst, 
  LanguageModel, 
  LoggerProtocol, 
  RolloutOutput, 
  Trajectory 
} from './types/index.js';
import { GEPAAdapter } from './core/adapter.js';
import { GEPAEngine } from './core/engine.js';
import { GEPAResult } from './core/result.js';
import { DefaultAdapter } from './adapters/default-adapter.js';
import { StdOutLogger } from './logging/logger.js';
import { ReflectiveMutationProposer } from './proposer/reflective-mutation.js';
import { MergeProposer } from './proposer/merge.js';
import { 
  ParetoCandidateSelector, 
  CurrentBestCandidateSelector 
} from './strategies/candidate-selector.js';
import { RoundRobinReflectionComponentSelector } from './strategies/component-selector.js';
import { EpochShuffledBatchSampler } from './strategies/batch-sampler.js';

export interface OptimizeConfig<D = DataInst> {
  seedCandidate: ComponentMap;
  trainset: D[];
  valset?: D[];
  adapter?: GEPAAdapter<D>;
  taskLM?: string | ((prompt: string) => Promise<string> | string);
  
  // Reflection configuration
  reflectionLM?: LanguageModel | string;
  candidateSelectionStrategy?: 'pareto' | 'current-best';
  skipPerfectScore?: boolean;
  reflectionMinibatchSize?: number;
  perfectScore?: number;
  
  // Merge configuration
  useMerge?: boolean;
  maxMergeInvocations?: number;
  
  // Budget
  maxMetricCalls: number;
  
  // Logging
  logger?: LoggerProtocol;
  runDir: string | null;
  useWandB?: boolean;
  wandBApiKey?: string;
  wandBInitKwargs?: Record<string, any>;
  trackBestOutputs?: boolean;
  displayProgressBar?: boolean;
  
  // Reproducibility
  seed?: number;
  raiseOnException?: boolean;
}

export async function optimize<D = DataInst>(
  config: OptimizeConfig<D>
): Promise<GEPAResult> {
  const {
    seedCandidate,
    trainset,
    valset = trainset,
    maxMetricCalls,
    candidateSelectionStrategy = 'pareto',
    skipPerfectScore = true,
    reflectionMinibatchSize = 3,
    perfectScore = 1,
    useMerge = false,
    maxMergeInvocations = 5,
    seed = 0,
    raiseOnException = true,
    trackBestOutputs = false,
    displayProgressBar = false
  } = config;
  
  let { adapter, taskLM, reflectionLM, logger } = config;
  const { runDir = null } = config;
  
  if (!adapter) {
    if (!taskLM) {
      throw new Error(
        'Since no adapter is provided, GEPA requires a task LM to be provided. Please set the `taskLM` parameter.'
      );
    }
    adapter = new DefaultAdapter(taskLM) as any;
  } else if (taskLM) {
    throw new Error(
      'Since an adapter is provided, GEPA does not require a task LM. Please set the `taskLM` parameter to undefined.'
    );
  }
  
  if (adapter && !adapter.proposeNewTexts && !reflectionLM) {
    throw new Error(
      'reflectionLM was not provided. The adapter used does not provide a proposeNewTexts method, ' +
      'and hence, GEPA will use the default proposer, which requires a reflectionLM to be specified.'
    );
  }
  
  if (typeof reflectionLM === 'string') {
    const modelName = reflectionLM;
    reflectionLM = async (prompt: string) => {
      const { OpenAI } = await import('openai');
      const openai = new OpenAI();
      const response = await openai.chat.completions.create({
        model: modelName,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.7
      });
      return response.choices[0].message.content || '';
    };
  }
  
  logger = logger || new StdOutLogger();
  
  let seedValue = seed;
  const rng = () => {
    const x = Math.sin(seedValue++) * 10000;
    return x - Math.floor(x);
  };
  
  const candidateSelector = candidateSelectionStrategy === 'pareto' ? 
    new ParetoCandidateSelector(rng) : 
    new CurrentBestCandidateSelector();
  
  const componentSelector = new RoundRobinReflectionComponentSelector();
  const batchSampler = new EpochShuffledBatchSampler<D>(reflectionMinibatchSize, rng);
  
  const reflectiveProposer = new ReflectiveMutationProposer<D>({
    logger,
    trainset,
    adapter: adapter as GEPAAdapter<D>,
    candidateSelector,
    componentSelector,
    batchSampler,
    perfectScore,
    skipPerfectScore,
    useWandB: config.useWandB,
    reflectionLM: reflectionLM as LanguageModel
  });
  
  const evaluator = (inputs: D[], prog: ComponentMap) => {
    const evalOut = (adapter as GEPAAdapter<D>).evaluate(inputs, prog, false);
    if (evalOut instanceof Promise) {
      return evalOut.then(result => [result.outputs, result.scores] as [RolloutOutput[], number[]]);
    }
    return [evalOut.outputs, evalOut.scores] as [RolloutOutput[], number[]];
  };
  
  let mergeProposer: MergeProposer<D> | undefined;
  if (useMerge) {
    mergeProposer = new MergeProposer<D>({
      logger,
      valset,
      evaluator,
      useMerge,
      maxMergeInvocations,
      rng
    });
  }
  
  const engine = new GEPAEngine<D>({
    runDir,
    evaluator,
    valset,
    seedCandidate,
    maxMetricCalls,
    perfectScore,
    seed,
    reflectiveProposer,
    mergeProposer,
    logger,
    useWandB: config.useWandB,
    wandBConfig: {
      apiKey: config.wandBApiKey,
      initKwargs: config.wandBInitKwargs
    },
    trackBestOutputs,
    displayProgressBar,
    raiseOnException
  });
  
  const state = await engine.run();
  return GEPAResult.fromState(state);
}