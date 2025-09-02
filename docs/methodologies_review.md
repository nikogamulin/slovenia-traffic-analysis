# Methodologies for Traffic Ripple Effect Analysis
## Slovenian Highway Network Investigation

### Executive Summary
This document reviews four advanced analytical methodologies for quantifying traffic ripple effects in the Slovenian highway network. Each methodology offers unique insights into how localized disruptions propagate through the transportation system, particularly during the 2025 traffic collapse period.

---

## 1. Network Graph Analysis

### Core Principle
Network graph analysis models the road system as a directed graph where intersections/interchanges are nodes and road segments are edges. This approach quantifies the structural importance of each segment and identifies critical bottlenecks that amplify congestion propagation.

### Application to Slovenian Network
- **Graph Construction**: Build from the Slovenian Road Segment Analysis, with 40+ major interchanges and 100+ segments
- **Edge Weights**: Assign based on segment capacity, typical flow volumes, and real-time speed data
- **Key Metrics**:
  - **Betweenness Centrality**: Identifies segments that lie on the shortest paths between multiple origin-destination pairs
  - **Flow-weighted Centrality**: Adjusts for actual traffic volumes, not just topology
  - **Vulnerability Index**: Measures network performance degradation when specific segments are impaired

### Implementation Approach
```python
# Pseudo-code structure
G = nx.DiGraph()
# Add nodes (interchanges): Kozarje, Malence, Slivnica, etc.
# Add weighted edges based on segment characteristics
# Calculate centrality metrics
betweenness = nx.betweenness_centrality(G, weight='flow_volume')
# Simulate segment disruptions
vulnerability = simulate_segment_removal(G, critical_segments)
```

### Strengths
- Provides system-wide perspective on network vulnerabilities
- Identifies non-obvious critical segments that may not carry highest volumes but are structurally crucial
- Enables "what-if" scenario analysis for infrastructure planning

### Limitations
- Static representation may not capture temporal dynamics
- Requires accurate OD matrix estimation for flow-weighted metrics
- May oversimplify driver route choice behavior

---

## 2. Spatiotemporal Vector Autoregression (VAR)

### Core Principle
VAR models capture the dynamic interdependencies between multiple time series. In traffic analysis, they quantify how speed/volume changes at one location predict future changes at other locations, revealing the temporal and spatial propagation of congestion.

### Application to Slovenian Network
- **Variables**: Hourly average speeds at 20 monitoring points
- **Spatial Structure**: Order monitoring points by network distance and flow direction
- **Lag Structure**: Test lags from 15 minutes to 3 hours to capture both immediate and delayed effects
- **Granger Causality**: Identify which monitoring points are "leading indicators" of downstream congestion

### Mathematical Formulation
```
Y_t = A_1*Y_{t-1} + A_2*Y_{t-2} + ... + A_p*Y_{t-p} + ε_t

Where:
Y_t = Vector of speeds at all monitoring points at time t
A_i = Coefficient matrices capturing cross-location dependencies
p = Optimal lag order (determined by AIC/BIC)
```

### Implementation Approach
```python
from statsmodels.tsa.api import VAR
# Prepare multivariate time series
speed_matrix = prepare_speed_data(monitoring_points, hourly)
# Fit VAR model
model = VAR(speed_matrix)
results = model.fit(maxlags=12, ic='aic')
# Impulse Response Analysis
irf = results.irf(periods=24)
# Identify propagation patterns
granger_matrix = results.test_causality()
```

### Strengths
- Captures complex temporal dependencies between locations
- Provides quantitative estimates of speed propagation magnitude and timing
- Can identify feedback loops and bidirectional effects

### Limitations
- Assumes linear relationships between variables
- Requires stationarity (may need differencing/detrending)
- High dimensionality with 20+ monitoring points may require dimension reduction

---

## 3. Interrupted Time Series (ITS) Analysis

### Core Principle
ITS is a quasi-experimental design that isolates the causal impact of an intervention (roadwork start) by comparing pre- and post-intervention trends while controlling for underlying patterns, seasonality, and confounders.

### Application to Slovenian Network
- **Intervention Points**: Start dates of major roadwork projects (e.g., January 1, 2024 for A1 Slovenske Konjice)
- **Outcome Variables**: Average speed and traffic volume at affected segments
- **Control Variables**: Day of week, hour of day, holidays, weather conditions
- **Counterfactual Construction**: Use unaffected segments as control group or synthetic control method

### Statistical Model
```
Y_t = β_0 + β_1*Time_t + β_2*Intervention_t + β_3*Time_after_t + 
      β_4*Season_t + β_5*Holiday_t + β_6*Weather_t + ε_t

Where:
β_2 = Immediate impact of roadwork (level change)
β_3 = Change in trend after roadwork starts (slope change)
```

### Implementation Approach
```python
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposed = seasonal_decompose(speed_series, model='additive')
# Create intervention variables
data['intervention'] = (data['date'] >= roadwork_start).astype(int)
data['time_after'] = np.maximum(0, (data['date'] - roadwork_start).dt.days)
# Fit segmented regression
model = sm.OLS(speed ~ time + intervention + time_after + controls)
results = model.fit()
# Calculate effect sizes with confidence intervals
immediate_effect = results.params['intervention']
trend_change = results.params['time_after']
```

### Strengths
- Provides causal estimates under reasonable assumptions
- Can isolate effects of specific interventions from background trends
- Allows for both immediate and gradual impact assessment

### Limitations
- Requires clear intervention timing (may be fuzzy for phased roadworks)
- Assumes no simultaneous interventions affecting outcome
- May be biased if drivers anticipate roadworks and change behavior pre-emptively

---

## 4. Bayesian Network Modeling

### Core Principle
Bayesian networks represent probabilistic dependencies between events using directed acyclic graphs (DAGs). They quantify how the probability of congestion changes given various combinations of conditions (roadworks, weather, holidays).

### Application to Slovenian Network
- **Network Structure**: 
  - Parent nodes: Roadwork status, weather conditions, holiday periods, time of day
  - Intermediate nodes: Incident occurrence, route choice changes
  - Child nodes: Congestion severity at key segments
- **Probability Estimation**: Learn conditional probability tables from historical data
- **Inference**: Calculate P(Severe_Congestion | Evidence) for various scenarios

### Network Architecture
```
Weather → Incidents → Congestion
   ↓                      ↑
Visibility            Roadworks
                          ↑
                      Holidays → Route_Choice
```

### Implementation Approach
```python
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Define network structure
model = BayesianNetwork([
    ('roadwork', 'congestion'),
    ('weather', 'incidents'),
    ('incidents', 'congestion'),
    ('holidays', 'route_choice'),
    ('route_choice', 'congestion')
])
# Learn parameters from data
model.fit(data, estimator=MaximumLikelihoodEstimator)
# Perform inference
inference = VariableElimination(model)
# Query: P(congestion='severe' | roadwork='active', holiday='yes')
result = inference.query(['congestion'], 
                        evidence={'roadwork': 1, 'holiday': 1})
```

### Strengths
- Handles uncertainty and incomplete information naturally
- Can incorporate expert knowledge into network structure
- Provides interpretable probabilistic predictions
- Enables counterfactual reasoning ("what if" scenarios)

### Limitations
- Requires careful specification of network structure
- Assumes conditional independence given parents (may not hold)
- Computational complexity increases with network size
- May require discretization of continuous variables

---

## Integrated Approach for Comprehensive Analysis

### Recommended Implementation Strategy

1. **Phase 1 - Structural Analysis**: Use Network Graph Analysis to identify critical segments and establish baseline vulnerability metrics

2. **Phase 2 - Temporal Dynamics**: Apply VAR models to understand propagation patterns and time delays between monitoring points

3. **Phase 3 - Causal Attribution**: Employ ITS analysis for specific roadwork projects to quantify their individual impacts

4. **Phase 4 - Probabilistic Integration**: Build Bayesian Network to synthesize findings and predict congestion under various scenarios

### Cross-Validation Framework
- Use 2020-2023 data for model training
- Validate on 2024 data
- Test predictions on 2025 "collapse" period
- Employ rolling window validation for time series models

### Key Performance Metrics
- **Prediction Accuracy**: RMSE for speed predictions
- **Congestion Detection**: Precision/Recall for severe congestion events
- **Ripple Effect Quantification**: Spatial correlation decay functions
- **Intervention Impact**: Effect sizes with 95% confidence/credible intervals

---

## Conclusion

Each methodology offers unique insights into the traffic collapse phenomenon:
- **Network Graph Analysis** reveals structural vulnerabilities
- **VAR Models** capture dynamic propagation patterns
- **ITS Analysis** isolates causal impacts of specific interventions
- **Bayesian Networks** integrate multiple factors probabilistically

The combination of these approaches will provide a comprehensive understanding of how roadworks, combined with the network's structural vulnerabilities, led to the 2025 traffic collapse in Slovenia.