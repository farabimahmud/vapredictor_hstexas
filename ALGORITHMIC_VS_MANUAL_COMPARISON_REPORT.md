"""
ALGORITHMIC VS MANUAL CAUSAL ANALYSIS COMPARISON REPORT
Texas Youth Risk Behavior Survey - Vaping Behavior Study

EXECUTIVE SUMMARY
================================================================================

This report compares three approaches to causal analysis:
1. Manual Theory-Based Confounder Selection
2. Algorithmic Discovery + Cross-Validation + Sensitivity Analysis
3. Robustness Assessment Across Methods

METHODOLOGICAL COMPARISON
================================================================================

## MANUAL APPROACH (Previous Analysis)
### Confounder Selection Strategy:
- Fixed set: ['age', 'sex', 'grade', 'race4', 'q26', 'q32']
- Theory-driven based on epidemiological literature
- Same confounders for all treatments
- No data-driven validation

### Estimation Method:
- Single DoWhy estimation per treatment
- Linear regression backend
- Basic refutation testing
- No sensitivity analysis

## ALGORITHMIC APPROACH (New Analysis)
### Confounder Selection Strategy:
- Causal graph discovery using correlation-based methods
- Cross-validation optimization for each treatment
- Statistical significance testing
- Ensemble scoring combining multiple methods
- Treatment-specific confounder sets

### Estimation Method:
- Multiple estimation strategies:
  * Unadjusted (baseline)
  * Minimal adjustment (demographics only)
  * Algorithmic selection (optimized)
  * Full adjustment (all available confounders)
- Comprehensive sensitivity analysis
- Robustness assessment with confidence ratings

CAUSAL EFFECT COMPARISON
================================================================================

## MENTAL HEALTH ISSUES (q26)
Manual Approach:    +0.0716 (single estimate)
Algorithmic Range:  +0.0338 to +0.0725
Best Estimate:      +0.0424 (High Robustness)
Interpretation:     Algorithmic approach shows MORE CONSERVATIVE effect

## EVER CIGARETTE USE (q32)
Manual Approach:    +0.2289 (single estimate)
Algorithmic Range:  -0.0795 to +0.2349
Best Estimate:      -0.0795 (Low Robustness)
Interpretation:     MAJOR DISAGREEMENT - Manual shows strong risk, Algorithmic shows protection

## CURRENT ALCOHOL USE (q42)
Manual Approach:    -0.0509 (single estimate)
Algorithmic Range:  -0.0661 to -0.0280
Best Estimate:      -0.0344 (High Robustness)
Interpretation:     Both show protective effect, algorithmic more conservative

## EVER MARIJUANA USE (q46)
Manual Approach:    -0.0795 (single estimate)
Algorithmic Range:  -0.1042 to +0.0735
Best Estimate:      -0.0954 (Low Robustness)
Interpretation:     Agreement on direction, but low robustness indicates uncertainty

## ADEQUATE SLEEP (q85)
Manual Approach:    +0.2339 (single estimate)
Algorithmic Range:  +0.1181 to +0.3373
Best Estimate:      +0.1181 (Low Robustness)
Interpretation:     Both show risk factor, but effect less robust than manual suggested

KEY INSIGHTS FROM ALGORITHMIC ANALYSIS
================================================================================

## 1. HIGH CONFIDENCE FINDINGS (Robust Across Methods)
✓ Mental Health Issues: Consistent risk factor (+0.0424)
✓ Current Alcohol Use: Consistent protective effect (-0.0344)

## 2. LOW CONFIDENCE FINDINGS (Sensitive to Confounders)
⚠ Ever Cigarette Use: Effect direction varies dramatically
⚠ Ever Marijuana Use: Large confidence intervals
⚠ Adequate Sleep: Counterintuitive effect with low robustness

## 3. CONFOUNDER SELECTION INSIGHTS

### Treatment-Specific Optimization:
Each treatment now has optimized confounder sets:
- Mental Health: ['q27', 'q25', 'q14', 'q47', 'q10', 'q12']
- Cigarette Use: ['q25', 'q47', 'q10', 'q12', 'q49', 'q85']
- Alcohol Use: ['q27', 'q25', 'q14', 'q47', 'grade', 'q12']
- Marijuana Use: ['q27', 'q25', 'q14', 'q47', 'q10', 'grade']
- Sleep: ['q27', 'q25', 'q14', 'q10', 'q12', 'q49']

### Cross-Validation Scores:
All optimized confounder sets achieved CV scores > 0.94, indicating excellent predictive performance.

DISCOVERED CAUSAL STRUCTURE
================================================================================

## Algorithmic Discovery Results:
The correlation-based causal discovery identified 17 potential causal parents of vaping:
['q26', 'q32', 'q40', 'q42', 'q43', 'q46', 'q47', 'q49', 'q10', 'q24', 'q25', 'q85', 'q12', 'q13', 'q17', 'q18', 'q20']

## Interpretation:
- Mental health factors (q26, q27, q28, q29)
- Substance use behaviors (q32, q40, q42, q43, q46, q47, q49)
- Risk behaviors (q10, q24, q25)
- Additional behavioral indicators (q12, q13, q17, q18, q20)
- Sleep patterns (q85)

This comprehensive network suggests vaping is embedded in a complex web of adolescent risk behaviors.

METHODOLOGICAL ADVANTAGES OF ALGORITHMIC APPROACH
================================================================================

## 1. DATA-DRIVEN VALIDATION
✓ Cross-validation ensures confounders improve prediction
✓ Multiple methods provide robustness checks
✓ Sensitivity analysis reveals estimate stability

## 2. TREATMENT-SPECIFIC OPTIMIZATION
✓ Each treatment gets optimal confounder set
✓ Avoids one-size-fits-all approach
✓ Maximizes statistical power for each analysis

## 3. UNCERTAINTY QUANTIFICATION
✓ Robustness ratings (High/Medium/Low confidence)
✓ Effect ranges across different specifications
✓ Standard deviation of estimates across methods

## 4. BIAS DETECTION
✓ Identifies effects sensitive to model specification
✓ Reveals when manual approaches may be misleading
✓ Provides evidence for/against causal assumptions

REVISED EVIDENCE-BASED RECOMMENDATIONS
================================================================================

## TIER 1: HIGH CONFIDENCE INTERVENTIONS
### Mental Health Support (Effect: +0.0424, High Robustness)
- Strategy: Comprehensive mental health screening and counseling
- Evidence: Robust across all model specifications
- Priority: Immediate implementation

### Address Alcohol Co-occurrence (Effect: -0.0344, High Robustness)
- Strategy: Understanding alcohol-vaping substitution patterns
- Evidence: Consistently protective effect
- Priority: Further research on mechanisms

## TIER 2: MEDIUM CONFIDENCE INTERVENTIONS
None identified in current analysis.

## TIER 3: REQUIRES FURTHER INVESTIGATION
### Tobacco Prevention Programs
- Manual analysis suggested strong gateway effect (+0.2289)
- Algorithmic analysis shows opposite effect (-0.0795)
- Recommendation: Collect additional data, investigate confounding

### Sleep Hygiene Programs  
- Effect size varies dramatically (+0.1181 to +0.3373)
- Low robustness suggests unmeasured confounding
- Recommendation: Investigate social/lifestyle factors

### Marijuana Policy Considerations
- Protective effect but low robustness
- May reflect substitution rather than causation
- Recommendation: Longitudinal studies needed

LIMITATIONS AND FUTURE DIRECTIONS
================================================================================

## 1. ALGORITHMIC LIMITATIONS
- Correlation-based discovery may miss complex relationships
- Cross-sectional data limits causal inference
- Computational constraints limited to 10,000 sample size

## 2. RECOMMENDED IMPROVEMENTS
- Implement full PC algorithm with larger computational resources
- Collect longitudinal data for temporal causal relationships
- Include additional social and environmental variables

## 3. VALIDATION NEEDS
- External validation on independent datasets
- Experimental studies for high-confidence effects
- Mediation analysis for mechanism understanding

CONCLUSION
================================================================================

The algorithmic approach reveals that many effects identified through manual 
confounder selection are NOT ROBUST to different model specifications. Only 
mental health interventions show consistent causal effects across all methods.

This analysis demonstrates the critical importance of:
1. Algorithmic confounder selection over theory-only approaches
2. Comprehensive sensitivity analysis for all causal estimates  
3. Robustness assessment before making policy recommendations
4. Treatment-specific optimization rather than one-size-fits-all confounding

The field of adolescent substance use prevention would benefit from adopting
these more rigorous causal inference methodologies to avoid potentially
ineffective or misdirected interventions.

================================================================================
Report Generated: October 28, 2025
Analysis Method: Algorithmic Causal Discovery + Cross-Validation + Sensitivity Analysis
Sample Size: 10,000 (sampled from 32,324 total observations)
Confidence Assessment: Implemented across all estimates
================================================================================
"""