import { useMemo, useState } from 'react';
import {
  Background,
  Controls,
  MiniMap,
  ReactFlow,
  Handle,
  Position,
} from '@xyflow/react';

import type { Node, Edge } from '@xyflow/react';

type ArchitectureNodeData = {
  id: string;
  number: string;
  title: string;
  category: string;
  description: string;
  type: 'main' | 'research' | 'crosscutting' | 'mvp';
  status: 'MVP' | 'Future' | 'Both';
  trigger: string;
  frequency: string;
  modules: string[];
  entities: string[];
  procedures: string[];
  methods: string[];
  inputs: string[];
  outputs: string[];
  dependencies: string[];
  technology: string[];
};

const architecture: ArchitectureNodeData[] = [
  {
    id: 'boundary',
    number: '3.1',
    title: 'System Objectives & Architecture Boundaries',
    category: 'FOUNDATION',
    type: 'main',
    status: 'MVP',
    trigger: 'Milestone-driven',
    frequency: 'Architecture milestone',
    description:
      'Defines exactly what Project 1A builds, consumes, produces and deliberately excludes.',
    modules: ['Scope Definition', 'System Boundary', 'Consumer Definition'],
    entities: ['SystemObjective', 'SystemBoundary', 'SystemConsumer'],
    procedures: ['defineScope()', 'defineBoundary()', 'defineConsumers()'],
    methods: ['Scope analysis', 'Boundary analysis'],
    inputs: ['Business requirements', 'BFSI requirements'],
    outputs: ['Approved architecture scope'],
    dependencies: [],
    technology: ['Architecture documentation'],
  },

  {
    id: 'data',
    number: '3.2',
    title: 'External Data Architecture',
    category: 'DATA',
    type: 'main',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'Source-dependent',
    description:
      'Defines the external market, volatility, macroeconomic and cross-asset information entering the platform.',
    modules: [
      'Market Data Connector',
      'India VIX Connector',
      'Macro Connector',
      'Cross-Asset Connector',
    ],
    entities: [
      'MarketObservation',
      'VolatilityObservation',
      'MacroObservation',
    ],
    procedures: [
      'fetchMarketData()',
      'fetchVolatilityData()',
      'fetchMacroData()',
    ],
    methods: ['API ingestion', 'Historical retrieval', 'Source normalization'],
    inputs: [
      'NSE/index data',
      'India VIX',
      'Macro indicators',
      'Cross-asset data',
    ],
    outputs: ['Raw market observations'],
    dependencies: ['3.1'],
    technology: ['Python', 'Data connectors'],
  },

  {
    id: 'ingestion',
    number: '3.3',
    title: 'Data Ingestion Architecture',
    category: 'DATA',
    type: 'main',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'Whenever new data arrives',
    description:
      'Acquires external data, validates payloads and records immutable raw observations.',
    modules: [
      'Data Fetcher',
      'Schema Validator',
      'Ingestion Logger',
      'Raw Data Writer',
    ],
    entities: ['RawDataset', 'IngestionEvent', 'DataSource'],
    procedures: [
      'fetchData()',
      'validatePayload()',
      'normalizeSchema()',
      'storeRawData()',
    ],
    methods: ['Retry', 'Validation', 'Normalization', 'Provenance tracking'],
    inputs: ['External data'],
    outputs: ['Validated raw datasets'],
    dependencies: ['3.2'],
    technology: ['Python'],
  },

  {
    id: 'processing',
    number: '3.4',
    title: 'Data Processing Architecture',
    category: 'DATA',
    type: 'main',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'After ingestion',
    description:
      'Cleans, aligns and prepares raw datasets for feature generation.',
    modules: [
      'Missing Data Handler',
      'Timestamp Normalizer',
      'Frequency Aligner',
      'Outlier Detector',
      'Data Quality Engine',
    ],
    entities: ['CleanDataset', 'DataQualityReport'],
    procedures: [
      'validateSchema()',
      'handleMissingValues()',
      'normalizeTimestamps()',
      'alignFrequency()',
      'detectOutliers()',
    ],
    methods: ['Imputation', 'Filtering', 'Resampling', 'Quality scoring'],
    inputs: ['Raw datasets'],
    outputs: ['Clean aligned datasets'],
    dependencies: ['3.3'],
    technology: ['Python', 'Pandas', 'NumPy'],
  },

  {
    id: 'features',
    number: '3.5',
    title: 'Feature Engineering Architecture',
    category: 'ML',
    type: 'main',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'After data processing',
    description:
      'Transforms processed market and macro data into model-ready features.',
    modules: [
      'Return Engine',
      'Momentum Engine',
      'Volatility Engine',
      'Trend Engine',
      'Drawdown Engine',
      'Breadth Engine',
      'Correlation Engine',
      'Macro Feature Engine',
    ],
    entities: ['Feature', 'FeatureVector', 'FeatureVersion'],
    procedures: [
      'calculateReturns()',
      'calculateMomentum()',
      'calculateVolatility()',
      'calculateTrend()',
      'calculateDrawdown()',
      'calculateBreadth()',
      'calculateCorrelation()',
    ],
    methods: [
      'Rolling statistics',
      'Percentage change',
      'Z-score',
      'Moving averages',
      'Lookback windows',
    ],
    inputs: ['Clean datasets'],
    outputs: ['Feature matrix'],
    dependencies: ['3.4'],
    technology: ['Python', 'Pandas', 'NumPy'],
  },

  {
    id: 'regime',
    number: '3.6',
    title: 'Regime Detection Architecture',
    category: 'CORE INTELLIGENCE',
    type: 'main',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'Every valid feature update',
    description:
      'Core intelligence layer that detects the latent Indian equity market regime.',
    modules: [
      'HMM Model',
      'State Estimator',
      'Posterior Probability Engine',
      'Transition Matrix',
      'Regime Mapper',
      'Transition Detector',
    ],
    entities: ['RegimeState', 'RegimeProbability', 'RegimeTransition'],
    procedures: [
      'fit()',
      'predict()',
      'predictProba()',
      'detectTransition()',
      'mapLatentState()',
      'getCurrentRegime()',
    ],
    methods: [
      'Forward algorithm',
      'Backward algorithm',
      'Viterbi decoding',
      'Posterior inference',
    ],
    inputs: ['Feature matrix'],
    outputs: [
      'Current regime',
      'Regime probabilities',
      'Transition probabilities',
    ],
    dependencies: ['3.5'],
    technology: ['Python', 'HMM', 'hmmlearn'],
  },

  {
    id: 'forecast',
    number: '3.7',
    title: 'Direction Forecasting Architecture',
    category: 'CORE INTELLIGENCE',
    type: 'main',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'After regime inference',
    description:
      'Forecasts future Indian equity-market direction using features and regime information.',
    modules: [
      'Target Builder',
      'Baseline Forecast Model',
      'Regime-Aware Forecast Model',
      'Probability Estimator',
    ],
    entities: ['Forecast', 'Prediction', 'ForecastTarget'],
    procedures: ['buildTarget()', 'train()', 'predict()', 'predictProba()'],
    methods: [
      'Logistic regression',
      'Tree-based models',
      'Regime-aware modelling',
    ],
    inputs: ['Features', 'Current regime', 'Regime probabilities'],
    outputs: ['Bullish/Bearish', 'Direction probability', 'Forecast horizon'],
    dependencies: ['3.5', '3.6'],
    technology: ['Python', 'scikit-learn'],
  },

  {
    id: 'ensemble',
    number: '3.8',
    title: 'Ensemble Architecture',
    category: 'FORECASTING',
    type: 'main',
    status: 'Future',
    trigger: 'Event-driven',
    frequency: 'After model predictions',
    description:
      'Combines multiple validated forecasting models and measures their agreement.',
    modules: [
      'Model Registry',
      'Ensemble Engine',
      'Weight Manager',
      'Agreement Calculator',
    ],
    entities: ['Model', 'ModelWeight', 'EnsemblePrediction'],
    procedures: [
      'combinePredictions()',
      'calculateWeights()',
      'calculateAgreement()',
    ],
    methods: ['Weighted averaging', 'Dynamic weighting', 'Model voting'],
    inputs: ['Individual model predictions'],
    outputs: ['Combined forecast', 'Model agreement'],
    dependencies: ['3.7'],
    technology: ['Python', 'ML frameworks'],
  },

  {
    id: 'uncertainty',
    number: '3.9',
    title: 'Uncertainty & Confidence Architecture',
    category: 'INTELLIGENCE',
    type: 'main',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'After every forecast',
    description:
      'Quantifies regime uncertainty, forecast uncertainty, model uncertainty and data uncertainty.',
    modules: [
      'Uncertainty Engine',
      'Confidence Engine',
      'Calibration Engine',
      'Probability Validator',
    ],
    entities: ['Uncertainty', 'ConfidenceScore', 'CalibrationResult'],
    procedures: [
      'calculateRegimeUncertainty()',
      'calculateForecastUncertainty()',
      'calculateConfidence()',
      'calibrateProbabilities()',
    ],
    methods: [
      'Entropy',
      'Probability calibration',
      'Model disagreement',
      'Conformal prediction',
    ],
    inputs: ['Regime probabilities', 'Forecast probabilities', 'Data quality'],
    outputs: ['Confidence', 'Uncertainty', 'Calibration'],
    dependencies: ['3.6', '3.7'],
    technology: ['Python', 'scikit-learn'],
  },

  {
    id: 'explainability',
    number: '3.10',
    title: 'Explainability Architecture',
    category: 'INTELLIGENCE',
    type: 'main',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'After every prediction',
    description:
      'Explains the major factors behind regime and direction predictions.',
    modules: [
      'Explanation Engine',
      'Feature Contribution Engine',
      'Regime Explanation Engine',
      'Forecast Explanation Engine',
    ],
    entities: ['Explanation', 'FeatureContribution', 'Driver'],
    procedures: [
      'generateExplanation()',
      'calculateFeatureContribution()',
      'explainRegime()',
      'explainForecast()',
    ],
    methods: [
      'Feature attribution',
      'Contribution ranking',
      'Historical comparison',
    ],
    inputs: ['Features', 'Regime', 'Forecast'],
    outputs: ['Major drivers', 'Human-readable explanation'],
    dependencies: ['3.5', '3.6', '3.7'],
    technology: ['Python'],
  },

  {
    id: 'monitoring',
    number: '3.11',
    title: 'Monitoring & MLOps Architecture',
    category: 'OPERATIONS',
    type: 'crosscutting',
    status: 'Both',
    trigger: 'Event-driven',
    frequency: 'Every ingestion/inference cycle',
    description:
      'Monitors data quality, model performance, regime behaviour and production health.',
    modules: [
      'Data Health Monitor',
      'Model Health Monitor',
      'Regime Monitor',
      'Drift Detector',
      'Alert Manager',
    ],
    entities: ['HealthMetric', 'DriftEvent', 'Alert'],
    procedures: [
      'checkDataHealth()',
      'checkModelHealth()',
      'checkRegimeHealth()',
      'detectDrift()',
      'raiseAlert()',
    ],
    methods: [
      'Distribution drift',
      'Performance monitoring',
      'Data quality monitoring',
    ],
    inputs: ['Pipeline metrics', 'Predictions', 'Outcomes'],
    outputs: ['Alerts', 'Drift signals', 'Health status'],
    dependencies: ['3.3', '3.6', '3.7'],
    technology: ['Monitoring stack'],
  },

  {
    id: 'explainability2',
    number: '3.12',
    title: 'Explainability Architecture',
    category: 'GOVERNANCE',
    type: 'crosscutting',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'Every prediction',
    description:
      'Cross-cutting explainability and decision transparency layer.',
    modules: [
      'Explanation Registry',
      'Driver Ranking',
      'Explanation Formatter',
    ],
    entities: ['ExplanationRecord', 'Driver'],
    procedures: ['storeExplanation()', 'retrieveExplanation()'],
    methods: ['Traceability', 'Explanation versioning'],
    inputs: ['Model outputs'],
    outputs: ['Auditable explanations'],
    dependencies: ['3.10'],
    technology: ['Python'],
  },

  {
    id: 'technology',
    number: '3.13',
    title: 'Technology Stack Definition',
    category: 'FOUNDATION',
    type: 'crosscutting',
    status: 'MVP',
    trigger: 'Milestone-driven',
    frequency: 'Architecture review',
    description:
      'Defines and governs the technology choices used throughout Project 1A.',
    modules: [
      'Python Runtime',
      'ML Runtime',
      'API Runtime',
      'Database',
      'Container Runtime',
    ],
    entities: ['TechnologyChoice', 'Dependency'],
    procedures: [
      'selectTechnology()',
      'validateCompatibility()',
      'reviewDependency()',
    ],
    methods: [
      'Technology evaluation',
      'Dependency management',
      'Compatibility analysis',
    ],
    inputs: ['Architecture requirements'],
    outputs: ['Approved technology stack'],
    dependencies: ['3.1'],
    technology: [
      'Python',
      'Pandas',
      'NumPy',
      'scikit-learn',
      'FastAPI',
      'PostgreSQL',
      'MLflow',
      'Docker',
      'Git',
    ],
  },

  {
    id: 'security',
    number: '3.14',
    title: 'Security & BFSI Controls Architecture',
    category: 'GOVERNANCE',
    type: 'crosscutting',
    status: 'MVP',
    trigger: 'Request-driven',
    frequency: 'Every protected operation',
    description:
      'Provides authentication, authorization, auditability, traceability and BFSI controls.',
    modules: [
      'Authentication',
      'Authorization',
      'Secrets Manager',
      'Audit Logger',
      'Access Control',
      'Rate Limiter',
    ],
    entities: ['User', 'AccessPolicy', 'AuditRecord', 'Secret'],
    procedures: [
      'authenticateRequest()',
      'authorizeRequest()',
      'validateInput()',
      'recordAuditEvent()',
    ],
    methods: [
      'Role-based access',
      'Encryption',
      'Audit logging',
      'Secret isolation',
    ],
    inputs: ['API requests', 'Data access', 'Model operations'],
    outputs: ['Authorized operations', 'Audit records'],
    dependencies: ['3.12', '3.13'],
    technology: ['Application security'],
  },

  {
    id: 'tradeoffs',
    number: '3.15',
    title: 'Architecture Trade-offs & Decision Records',
    category: 'GOVERNANCE',
    type: 'crosscutting',
    status: 'MVP',
    trigger: 'Milestone-driven',
    frequency: 'Whenever architecture changes',
    description:
      'Records architectural decisions, alternatives, trade-offs and consequences.',
    modules: ['ADR Registry', 'Decision Review', 'Trade-off Analysis'],
    entities: ['ArchitectureDecision', 'Alternative', 'Tradeoff'],
    procedures: [
      'createADR()',
      'evaluateAlternative()',
      'approveDecision()',
      'reviewDecision()',
    ],
    methods: ['Cost-benefit analysis', 'Risk analysis', 'Complexity analysis'],
    inputs: ['Architecture requirements', 'Research results'],
    outputs: ['Architecture Decision Records'],
    dependencies: ['3.13'],
    technology: ['Markdown', 'Git'],
  },

  {
    id: 'mvp',
    number: '3.16',
    title: 'MVP Architecture Definition',
    category: 'MVP BOUNDARY',
    type: 'mvp',
    status: 'MVP',
    trigger: 'Milestone-driven',
    frequency: 'Architecture freeze',
    description:
      'Defines the smallest scientifically valid architecture capable of testing whether regime information adds predictive value.',
    modules: [
      'Data Pipeline',
      'Feature Engine',
      'HMM Regime Engine',
      'Direction Model',
      'Confidence Engine',
      'API',
    ],
    entities: [
      'RegimeState',
      'RegimeProbability',
      'Forecast',
      'ConfidenceScore',
    ],
    procedures: [
      'ingest()',
      'process()',
      'generateFeatures()',
      'detectRegime()',
      'forecastDirection()',
      'calculateConfidence()',
    ],
    methods: ['HMM', 'Directional classification', 'Walk-forward validation'],
    inputs: ['Market data', 'Macro data', 'Volatility data'],
    outputs: ['Regime', 'Probabilities', 'Direction', 'Confidence'],
    dependencies: ['3.3', '3.5', '3.6', '3.7', '3.9', '3.12'],
    technology: ['Python', 'FastAPI', 'PostgreSQL', 'Docker'],
  },

  {
    id: 'storage',
    number: '3.17',
    title: 'Storage Architecture',
    category: 'PLATFORM',
    type: 'crosscutting',
    status: 'MVP',
    trigger: 'Event-driven',
    frequency: 'After pipeline execution',
    description:
      'Persists raw data, processed data, features, regimes, forecasts, models and experiments.',
    modules: [
      'Raw Data Repository',
      'Feature Repository',
      'Regime Repository',
      'Forecast Repository',
      'Model Repository',
      'Experiment Repository',
    ],
    entities: [
      'DatasetVersion',
      'FeatureVersion',
      'RegimeState',
      'Forecast',
      'ModelVersion',
      'Experiment',
    ],
    procedures: ['save()', 'load()', 'query()', 'version()'],
    methods: ['Relational storage', 'Columnar storage', 'Versioning'],
    inputs: ['Pipeline outputs'],
    outputs: ['Historical records'],
    dependencies: ['3.3', '3.6', '3.7'],
    technology: ['PostgreSQL', 'Parquet'],
  },

  {
    id: 'api',
    number: '3.18',
    title: 'API Architecture',
    category: 'PLATFORM',
    type: 'main',
    status: 'MVP',
    trigger: 'Request-driven',
    frequency: 'On API request',
    description:
      'Exposes current regime, forecasts, history and explanations to downstream consumers.',
    modules: [
      'Regime Service',
      'Forecast Service',
      'History Service',
      'Explanation Service',
      'Health Service',
    ],
    entities: ['APIRequest', 'APIResponse', 'HealthStatus'],
    procedures: [
      'GET /regime/current',
      'GET /forecast/current',
      'GET /regime/history',
      'GET /forecast/history',
      'GET /health',
    ],
    methods: ['REST', 'JSON', 'Request validation'],
    inputs: ['Consumer requests'],
    outputs: ['Structured API responses'],
    dependencies: ['3.17', '3.14'],
    technology: ['FastAPI', 'Pydantic'],
  },

  {
    id: 'backtest',
    number: '3.19',
    title: 'Backtesting Architecture',
    category: 'RESEARCH',
    type: 'research',
    status: 'MVP',
    trigger: 'Time-driven',
    frequency: 'Periodic research cycle',
    description:
      'Evaluates historical performance using chronological and walk-forward validation.',
    modules: [
      'Backtest Engine',
      'Walk-Forward Validator',
      'Training Window',
      'Test Window',
      'Metrics Engine',
    ],
    entities: [
      'BacktestRun',
      'TrainingWindow',
      'TestWindow',
      'EvaluationResult',
    ],
    procedures: [
      'runBacktest()',
      'runWalkForward()',
      'evaluatePredictions()',
      'aggregateMetrics()',
    ],
    methods: [
      'Walk-forward validation',
      'Expanding window',
      'Rolling window',
      'Point-in-time validation',
    ],
    inputs: ['Historical datasets', 'Model versions'],
    outputs: ['Performance metrics'],
    dependencies: ['3.17'],
    technology: ['Python', 'Pandas', 'NumPy'],
  },

  {
    id: 'experiments',
    number: '3.20',
    title: 'Experimentation Architecture',
    category: 'RESEARCH',
    type: 'research',
    status: 'MVP',
    trigger: 'Milestone-driven',
    frequency: 'Every experiment',
    description:
      'Tracks experiments, datasets, models, features, parameters and evaluation results.',
    modules: [
      'Experiment Tracker',
      'Metric Tracker',
      'Dataset Registry',
      'Feature Registry',
      'Model Registry',
    ],
    entities: [
      'Experiment',
      'ExperimentRun',
      'Metric',
      'ModelVersion',
      'DatasetVersion',
    ],
    procedures: [
      'createExperiment()',
      'logRun()',
      'logMetrics()',
      'registerModel()',
    ],
    methods: ['Experiment versioning', 'Metric tracking', 'Artifact tracking'],
    inputs: ['Backtest results', 'Model runs'],
    outputs: ['Reproducible experiment records'],
    dependencies: ['3.19'],
    technology: ['MLflow'],
  },

  {
    id: 'mlops',
    number: '3.21',
    title: 'MLOps & Model Lifecycle Architecture',
    category: 'OPERATIONS',
    type: 'research',
    status: 'Both',
    trigger: 'Milestone-driven',
    frequency: 'Condition-driven / scheduled',
    description:
      'Controls model validation, promotion, deployment, rollback and retirement.',
    modules: [
      'Model Validator',
      'Model Promoter',
      'Deployment Manager',
      'Rollback Manager',
      'Retraining Manager',
    ],
    entities: ['ModelVersion', 'Deployment', 'ModelStatus'],
    procedures: [
      'validateModel()',
      'promoteModel()',
      'deployModel()',
      'rollbackModel()',
      'triggerRetraining()',
    ],
    methods: ['Model approval', 'Version promotion', 'Rollback'],
    inputs: ['Validated experiments', 'Monitoring results'],
    outputs: ['Approved production model'],
    dependencies: ['3.20', '3.11'],
    technology: ['Docker', 'MLflow'],
  },

  {
    id: 'deployment',
    number: '3.22',
    title: 'Deployment Architecture',
    category: 'OPERATIONS',
    type: 'research',
    status: 'Both',
    trigger: 'Milestone-driven',
    frequency: 'Release cycle',
    description:
      'Packages and exposes validated system components for controlled execution.',
    modules: [
      'Inference Service',
      'API Service',
      'Database',
      'Container Runtime',
    ],
    entities: ['Deployment', 'Release', 'Environment'],
    procedures: ['build()', 'test()', 'deploy()', 'rollback()'],
    methods: ['Container deployment', 'Environment promotion'],
    inputs: ['Validated model', 'Application build'],
    outputs: ['Running service'],
    dependencies: ['3.21', '3.14'],
    technology: ['Docker', 'FastAPI'],
  },
];

const mainFlow = [
  'boundary',
  'data',
  'ingestion',
  'processing',
  'features',
  'regime',
  'forecast',
  'ensemble',
  'uncertainty',
  'explainability',
  'storage',
  'api',
];

const researchFlow = [
  'storage',
  'backtest',
  'experiments',
  'mlops',
  'deployment',
];

const edges: Edge[] = [
  ...mainFlow.slice(0, -1).map((source, index) => ({
    id: `main-${index}`,
    source,
    target: mainFlow[index + 1],
    animated: true,
    style: { stroke: '#38bdf8', strokeWidth: 2 },
  })),

  {
    id: 'research-1',
    source: 'storage',
    target: 'backtest',
    animated: true,
    style: { stroke: '#facc15', strokeWidth: 2 },
  },
  {
    id: 'research-2',
    source: 'backtest',
    target: 'experiments',
    animated: true,
    style: { stroke: '#facc15', strokeWidth: 2 },
  },
  {
    id: 'research-3',
    source: 'experiments',
    target: 'mlops',
    animated: true,
    style: { stroke: '#a3e635', strokeWidth: 2 },
  },
  {
    id: 'research-4',
    source: 'mlops',
    target: 'deployment',
    animated: true,
    style: { stroke: '#a3e635', strokeWidth: 2 },
  },
  {
    id: 'research-5',
    source: 'deployment',
    target: 'monitoring',
    animated: true,
    style: { stroke: '#a3e635', strokeWidth: 2 },
  },
  {
    id: 'monitoring-regime',
    source: 'monitoring',
    target: 'regime',
    animated: true,
    style: {
      stroke: '#a3e635',
      strokeWidth: 1.5,
      strokeDasharray: '6 4',
    },
  },

  {
    id: 'security-api',
    source: 'security',
    target: 'api',
    style: {
      stroke: '#4ade80',
      strokeWidth: 2,
      strokeDasharray: '6 4',
    },
  },
  {
    id: 'technology-system',
    source: 'technology',
    target: 'features',
    style: {
      stroke: '#c084fc',
      strokeWidth: 1.5,
      strokeDasharray: '6 4',
    },
  },
  {
    id: 'tradeoffs-system',
    source: 'tradeoffs',
    target: 'mvp',
    style: {
      stroke: '#f472b6',
      strokeWidth: 1.5,
      strokeDasharray: '6 4',
    },
  },
  {
    id: 'mvp-boundary',
    source: 'mvp',
    target: 'regime',
    style: {
      stroke: '#facc15',
      strokeWidth: 1.5,
      strokeDasharray: '6 4',
    },
  },
];

const colorMap: Record<string, string> = {
  FOUNDATION: '#38bdf8',
  DATA: '#60a5fa',
  ML: '#a78bfa',
  'CORE INTELLIGENCE': '#c084fc',
  FORECASTING: '#e879f9',
  INTELLIGENCE: '#f472b6',
  PLATFORM: '#fb923c',
  RESEARCH: '#facc15',
  OPERATIONS: '#a3e635',
  GOVERNANCE: '#4ade80',
  'MVP BOUNDARY': '#facc15',
};

function ArchitectureNode({ data }: { data: ArchitectureNodeData }) {
  const color = colorMap[data.category] || '#94a3b8';

  return (
    <div
      className="architecture-node"
      style={{
        borderColor: `${color}66`,
        boxShadow: `0 10px 35px ${color}10`,
      }}
    >
      <Handle type="target" position={Position.Top} />

      <div className="node-number" style={{ color }}>
        {data.number}
      </div>

      <div className="node-category" style={{ color }}>
        {data.category}
      </div>

      <div className="node-title">{data.title}</div>

      <div className="node-meta">
        <span className={`status ${data.status.toLowerCase()}`}>
          {data.status}
        </span>

        <span className="trigger">{data.trigger}</span>
      </div>

      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

const nodeTypes = {
  architecture: ArchitectureNode,
};

function DetailList({ title, values }: { title: string; values: string[] }) {
  if (!values?.length) return null;

  return (
    <div className="detail-section">
      <div className="detail-heading">{title}</div>

      {values.map((value, index) => (
        <div className="detail-value" key={`${value}-${index}`}>
          {value}
        </div>
      ))}
    </div>
  );
}

export default function App() {
  const [selected, setSelected] = useState<ArchitectureNodeData | null>(null);

  const [search, setSearch] = useState('');

  const [view, setView] = useState<'combined' | 'main' | 'research' | 'mvp'>(
    'combined'
  );

  const filteredArchitecture = useMemo(() => {
    const q = search.toLowerCase().trim();

    return architecture.filter((item) => {
      if (view === 'main' && item.type !== 'main') return false;

      if (view === 'research' && item.type !== 'research') return false;

      if (view === 'mvp' && item.status === 'Future') return false;

      if (!q) return true;

      return [
        item.number,
        item.title,
        item.category,
        item.description,
        ...item.modules,
        ...item.entities,
        ...item.procedures,
        ...item.methods,
      ]
        .join(' ')
        .toLowerCase()
        .includes(q);
    });
  }, [search, view]);

  const visibleIds = new Set(filteredArchitecture.map((item) => item.id));

  const nodes: Node[] = filteredArchitecture.map((item) => {
    const isMain = mainFlow.includes(item.id);

    const mainIndex = mainFlow.indexOf(item.id);

    const researchIndex = researchFlow.indexOf(item.id);

    let position = {
      x: 0,
      y: 0,
    };

    if (isMain) {
      position = {
        x: 250,
        y: mainIndex * 145,
      };
    }

    if (item.type === 'research') {
      position = {
        x: 700,
        y: researchIndex >= 0 ? researchIndex * 155 + 250 : 0,
      };
    }

    if (item.id === 'monitoring') {
      position = {
        x: 1100,
        y: 730,
      };
    }

    if (item.id === 'security') {
      position = {
        x: -260,
        y: 600,
      };
    }

    if (item.id === 'technology') {
      position = {
        x: -260,
        y: 120,
      };
    }

    if (item.id === 'tradeoffs') {
      position = {
        x: -260,
        y: 850,
      };
    }

    if (item.id === 'mvp') {
      position = {
        x: 700,
        y: 1500,
      };
    }

    if (item.id === 'explainability2') {
      position = {
        x: 1100,
        y: 350,
      };
    }

    return {
      id: item.id,
      type: 'architecture',
      position,
      data: item,
    };
  });

  const visibleEdges = edges.filter(
    (edge) => visibleIds.has(edge.source) && visibleIds.has(edge.target)
  );

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <div className="brand">
            ZETHETA <span>1A</span>
          </div>

          <div className="subtitle">Unified Solution Architecture</div>
        </div>

        <div className="header-title">
          MARKET REGIME DETECTION & EQUITY DIRECTION FORECASTING
        </div>

        <div className="phase">
          PHASE 3<span>ARCHITECTURE COMPLETE</span>
        </div>
      </header>

      <div className="toolbar">
        <div className="views">
          <button
            className={view === 'combined' ? 'active' : ''}
            onClick={() => setView('combined')}
          >
            Combined
          </button>

          <button
            className={view === 'main' ? 'active' : ''}
            onClick={() => setView('main')}
          >
            Main Pipeline
          </button>

          <button
            className={view === 'research' ? 'active' : ''}
            onClick={() => setView('research')}
          >
            Research / MLOps
          </button>

          <button
            className={view === 'mvp' ? 'active' : ''}
            onClick={() => setView('mvp')}
          >
            MVP
          </button>
        </div>

        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search architecture..."
        />
      </div>

      <main className="workspace">
        <div className="flow">
          <ReactFlow
            nodes={nodes}
            edges={visibleEdges}
            nodeTypes={nodeTypes}
            onNodeClick={(_, node) =>
              setSelected(node.data as ArchitectureNodeData)
            }
            fitView
            fitViewOptions={{
              padding: 0.15,
            }}
          >
            <Background color="#1e293b" gap={28} size={1} />

            <Controls />

            <MiniMap
              nodeColor={(node) =>
                colorMap[(node.data as ArchitectureNodeData)?.category] ||
                '#64748b'
              }
            />
          </ReactFlow>

          <div className="flow-label">PROJECT 1A · UNIFIED ARCHITECTURE</div>
        </div>

        <aside className="panel">
          {selected ? (
            <>
              <button className="close" onClick={() => setSelected(null)}>
                ×
              </button>

              <div
                className="panel-number"
                style={{
                  color: colorMap[selected.category],
                }}
              >
                {selected.number}
              </div>

              <div
                className="panel-category"
                style={{
                  color: colorMap[selected.category],
                }}
              >
                {selected.category}
              </div>

              <h2>{selected.title}</h2>

              <div className="panel-badges">
                <span className={`status ${selected.status.toLowerCase()}`}>
                  {selected.status}
                </span>

                <span className="trigger">{selected.trigger}</span>
              </div>

              <p className="description">{selected.description}</p>

              <div className="execution">
                <strong>Execution Frequency</strong>
                <span>{selected.frequency}</span>
              </div>

              <DetailList title="Modules" values={selected.modules} />

              <DetailList title="Entities" values={selected.entities} />

              <DetailList title="Procedures" values={selected.procedures} />

              <DetailList title="Methods" values={selected.methods} />

              <DetailList title="Inputs" values={selected.inputs} />

              <DetailList title="Outputs" values={selected.outputs} />

              <DetailList title="Dependencies" values={selected.dependencies} />

              <DetailList title="Technology" values={selected.technology} />
            </>
          ) : (
            <div className="overview">
              <div className="overview-kicker">UNIFIED ARCHITECTURE</div>

              <h2>Project 1A</h2>

              <p>
                One connected architecture covering the complete market regime
                detection and equity direction forecasting system.
              </p>

              <div className="architecture-stats">
                <div>
                  <strong>{architecture.length}</strong>
                  <span>Architecture Components</span>
                </div>

                <div>
                  <strong>
                    {
                      architecture.filter(
                        (a) => a.status === 'MVP' || a.status === 'Both'
                      ).length
                    }
                  </strong>
                  <span>MVP / Both</span>
                </div>
              </div>

              <div className="legend">
                <div>
                  <span className="dot cyan" />
                  Main prediction pipeline
                </div>

                <div>
                  <span className="dot yellow" />
                  Research / backtesting
                </div>

                <div>
                  <span className="dot green" />
                  MLOps / operations
                </div>

                <div>
                  <span className="dot purple" />
                  Cross-cutting architecture
                </div>
              </div>

              <div className="instruction">
                <strong>Click any architecture node</strong>
                <br />
                Inspect its modules, entities, procedures, methods, inputs,
                outputs, dependencies and execution frequency.
              </div>

              <div className="principle">
                <span>ARCHITECTURAL PRINCIPLE</span>
                Build a modular regime-detection engine first, then
                progressively plug in more sophisticated models without
                redesigning the entire system.
              </div>
            </div>
          )}
        </aside>
      </main>

      <footer>
        ZETHETA PROJECT 1A
        <span>•</span>
        SOLUTION ARCHITECTURE
        <span>•</span>
        3.1 → 3.16 + SYSTEM COMPONENTS
      </footer>
    </div>
  );
}
