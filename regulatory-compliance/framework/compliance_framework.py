"""
Regulatory Compliance Framework
Core regulatory compliance system for institutional trading operations
Supports MiFID II, EMIR, US regulatory requirements (Dodd-Frank, etc.)
"""

import sys
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import json
import logging
import asyncio
from pathlib import Path

import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RegulatoryRegime(Enum):
    """Supported regulatory regimes"""
    MIFID_II = "MIFID_II"           # EU Markets in Financial Instruments Directive
    EMIR = "EMIR"                   # European Market Infrastructure Regulation
    DODD_FRANK = "DODD_FRANK"       # US Dodd-Frank Act
    CFTC = "CFTC"                   # US Commodity Futures Trading Commission
    SEC = "SEC"                     # US Securities and Exchange Commission
    FINRA = "FINRA"                 # Financial Industry Regulatory Authority
    FCA = "FCA"                     # UK Financial Conduct Authority
    ASIC = "ASIC"                   # Australian Securities and Investments Commission

class ReportingFrequency(Enum):
    """Reporting frequency options"""
    REAL_TIME = "REAL_TIME"         # Immediate reporting
    DAILY = "DAILY"                 # End of day
    WEEKLY = "WEEKLY"               # Weekly summary
    MONTHLY = "MONTHLY"             # Monthly reports
    QUARTERLY = "QUARTERLY"         # Quarterly filings
    ANNUAL = "ANNUAL"               # Annual reports

class ComplianceStatus(Enum):
    """Compliance check status"""
    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PENDING_REVIEW = "PENDING_REVIEW"
    EXCEPTION_APPROVED = "EXCEPTION_APPROVED"
    REQUIRES_ATTENTION = "REQUIRES_ATTENTION"

@dataclass
class RegulatoryEntity:
    """Regulatory entity information"""
    entity_id: str
    entity_name: str
    entity_type: str  # "INVESTMENT_FIRM", "FUND", "HEDGE_FUND", etc.
    regulatory_regimes: List[RegulatoryRegime]
    lei_code: Optional[str] = None  # Legal Entity Identifier
    regulatory_numbers: Dict[str, str] = field(default_factory=dict)
    contact_info: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ComplianceRule:
    """Individual compliance rule definition"""
    rule_id: str
    rule_name: str
    regulatory_regime: RegulatoryRegime
    rule_type: str  # "POSITION_LIMIT", "REPORTING", "RISK_LIMIT", etc.
    description: str
    parameters: Dict[str, Any]
    severity: str = "MEDIUM"  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    active: bool = True
    effective_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None

@dataclass
class ComplianceViolation:
    """Compliance violation record"""
    violation_id: str
    rule_id: str
    entity_id: str
    violation_type: str
    description: str
    severity: str
    detected_time: datetime
    resolved_time: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    status: ComplianceStatus = ComplianceStatus.REQUIRES_ATTENTION
    financial_impact: Optional[float] = None
    regulatory_impact: Optional[str] = None

@dataclass
class RegulatoryReport:
    """Regulatory report structure"""
    report_id: str
    report_type: str
    regulatory_regime: RegulatoryRegime
    entity_id: str
    reporting_period: Tuple[datetime, datetime]
    generated_time: datetime
    file_path: Optional[str] = None
    submission_status: str = "PENDING"
    submission_time: Optional[datetime] = None
    acknowledgment_received: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

class RegulatoryFramework:
    """
    Core regulatory compliance framework
    Manages compliance rules, monitoring, and reporting across multiple jurisdictions
    """
    
    def __init__(self, entity: RegulatoryEntity, config_path: str = None):
        """
        Initialize regulatory framework
        
        Args:
            entity: Regulatory entity information
            config_path: Path to regulatory configuration file
        """
        self.entity = entity
        self.config_path = config_path or "config/regulatory_config.json"
        
        # Core components
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.violations: List[ComplianceViolation] = []
        self.reports: Dict[str, RegulatoryReport] = {}
        
        # Monitoring state
        self.monitoring_active = False
        self.last_check_time = datetime.now(timezone.utc)
        
        # Load configuration
        self._load_regulatory_config()
        
        logger.info(f"Regulatory framework initialized for {entity.entity_name}")
        logger.info(f"Active regimes: {[regime.value for regime in entity.regulatory_regimes]}")
    
    def _load_regulatory_config(self):
        """Load regulatory configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                
                # Load compliance rules
                for rule_data in config.get('compliance_rules', []):
                    rule = ComplianceRule(
                        rule_id=rule_data['rule_id'],
                        rule_name=rule_data['rule_name'],
                        regulatory_regime=RegulatoryRegime(rule_data['regulatory_regime']),
                        rule_type=rule_data['rule_type'],
                        description=rule_data['description'],
                        parameters=rule_data['parameters'],
                        severity=rule_data.get('severity', 'MEDIUM'),
                        active=rule_data.get('active', True)
                    )
                    self.compliance_rules[rule.rule_id] = rule
                
                logger.info(f"Loaded {len(self.compliance_rules)} compliance rules")
            else:
                logger.warning(f"Regulatory config file not found: {self.config_path}")
                self._create_default_rules()
                
        except Exception as e:
            logger.error(f"Error loading regulatory config: {str(e)}")
            self._create_default_rules()
    
    def _create_default_rules(self):
        """Create default compliance rules for common requirements"""
        default_rules = [
            # MiFID II Rules
            ComplianceRule(
                rule_id="MIFID_POSITION_LIMIT",
                rule_name="MiFID II Position Limits",
                regulatory_regime=RegulatoryRegime.MIFID_II,
                rule_type="POSITION_LIMIT",
                description="Position limits for commodity derivatives under MiFID II",
                parameters={
                    "position_limit_pct": 25.0,  # 25% of deliverable supply
                    "monitoring_frequency": "DAILY",
                    "exemptions": ["hedging", "liquidity_provision"]
                },
                severity="HIGH"
            ),
            
            # EMIR Rules
            ComplianceRule(
                rule_id="EMIR_CLEARING_OBLIGATION",
                rule_name="EMIR Clearing Obligation",
                regulatory_regime=RegulatoryRegime.EMIR,
                rule_type="CLEARING",
                description="Central clearing obligation for standardized derivatives",
                parameters={
                    "clearing_threshold": 3000000000,  # €3bn notional
                    "asset_classes": ["IRS", "CDS", "FX_derivatives"],
                    "exemptions": ["pension_schemes", "non_financial"]
                },
                severity="CRITICAL"
            ),
            
            # US Dodd-Frank
            ComplianceRule(
                rule_id="DODD_FRANK_VOLCKER",
                rule_name="Volcker Rule Compliance",
                regulatory_regime=RegulatoryRegime.DODD_FRANK,
                rule_type="PROPRIETARY_TRADING",
                description="Volcker Rule proprietary trading restrictions",
                parameters={
                    "permitted_activities": ["market_making", "hedging", "government_securities"],
                    "metrics_threshold": 100000000,  # $100M
                    "monitoring_frequency": "DAILY"
                },
                severity="CRITICAL"
            ),
            
            # CFTC Position Limits
            ComplianceRule(
                rule_id="CFTC_POSITION_LIMITS",
                rule_name="CFTC Position Limits",
                regulatory_regime=RegulatoryRegime.CFTC,
                rule_type="POSITION_LIMIT",
                description="CFTC federal position limits for agricultural and energy commodities",
                parameters={
                    "spot_month_limit": 25.0,  # 25% of estimated deliverable supply
                    "non_spot_limit": 10.0,    # 10% of open interest
                    "exemptions": ["bona_fide_hedging"]
                },
                severity="HIGH"
            )
        ]
        
        for rule in default_rules:
            self.compliance_rules[rule.rule_id] = rule
        
        logger.info(f"Created {len(default_rules)} default compliance rules")
    
    def add_compliance_rule(self, rule: ComplianceRule) -> bool:
        """
        Add new compliance rule
        
        Args:
            rule: Compliance rule to add
            
        Returns:
            True if rule added successfully
        """
        try:
            # Validate rule applies to entity's regulatory regimes
            if rule.regulatory_regime not in self.entity.regulatory_regimes:
                logger.warning(f"Rule {rule.rule_id} regime {rule.regulatory_regime} not applicable to entity")
                return False
            
            self.compliance_rules[rule.rule_id] = rule
            logger.info(f"Added compliance rule: {rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding compliance rule: {str(e)}")
            return False
    
    def check_compliance(self, trading_data: Dict[str, Any]) -> List[ComplianceViolation]:
        """
        Check compliance against all active rules
        
        Args:
            trading_data: Current trading positions, transactions, risk metrics
            
        Returns:
            List of compliance violations found
        """
        violations = []
        
        try:
            for rule_id, rule in self.compliance_rules.items():
                if not rule.active:
                    continue
                
                # Check if rule applies to current data
                violation = self._check_rule_compliance(rule, trading_data)
                if violation:
                    violations.append(violation)
            
            # Store violations
            self.violations.extend(violations)
            
            if violations:
                logger.warning(f"Found {len(violations)} compliance violations")
            else:
                logger.debug("All compliance checks passed")
            
            return violations
            
        except Exception as e:
            logger.error(f"Error during compliance check: {str(e)}")
            return []
    
    def _check_rule_compliance(self, rule: ComplianceRule, trading_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """
        Check compliance for a specific rule
        
        Args:
            rule: Compliance rule to check
            trading_data: Current trading data
            
        Returns:
            ComplianceViolation if violation found, None otherwise
        """
        try:
            # Route to specific rule type handlers
            if rule.rule_type == "POSITION_LIMIT":
                return self._check_position_limits(rule, trading_data)
            elif rule.rule_type == "RISK_LIMIT":
                return self._check_risk_limits(rule, trading_data)
            elif rule.rule_type == "REPORTING":
                return self._check_reporting_requirements(rule, trading_data)
            elif rule.rule_type == "CLEARING":
                return self._check_clearing_obligations(rule, trading_data)
            elif rule.rule_type == "PROPRIETARY_TRADING":
                return self._check_proprietary_trading(rule, trading_data)
            else:
                logger.warning(f"Unknown rule type: {rule.rule_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error checking rule {rule.rule_id}: {str(e)}")
            return None
    
    def _check_position_limits(self, rule: ComplianceRule, trading_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check position limit compliance"""
        try:
            positions = trading_data.get('positions', {})
            market_data = trading_data.get('market_data', {})
            
            limit_pct = rule.parameters.get('position_limit_pct', 25.0)
            
            for symbol, position in positions.items():
                # Get market information for position sizing
                market_info = market_data.get(symbol, {})
                open_interest = market_info.get('open_interest', 1000000)  # Default if not available
                
                # Calculate position as percentage of open interest
                position_pct = abs(position.get('quantity', 0)) / open_interest * 100
                
                if position_pct > limit_pct:
                    return ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        entity_id=self.entity.entity_id,
                        violation_type="POSITION_LIMIT_EXCEEDED",
                        description=f"Position in {symbol} ({position_pct:.2f}%) exceeds limit ({limit_pct:.2f}%)",
                        severity=rule.severity,
                        detected_time=datetime.now(timezone.utc),
                        financial_impact=position.get('market_value', 0),
                        regulatory_impact="Potential regulatory action and fines"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking position limits: {str(e)}")
            return None
    
    def _check_risk_limits(self, rule: ComplianceRule, trading_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check risk limit compliance"""
        try:
            risk_metrics = trading_data.get('risk_metrics', {})
            
            # Check various risk limits based on rule parameters
            for metric, limit in rule.parameters.items():
                if metric in risk_metrics:
                    current_value = risk_metrics[metric]
                    
                    if isinstance(limit, dict):
                        # Complex limit structure
                        max_limit = limit.get('max')
                        min_limit = limit.get('min')
                        
                        if max_limit and current_value > max_limit:
                            return self._create_risk_violation(rule, metric, current_value, max_limit, "EXCEEDED")
                        elif min_limit and current_value < min_limit:
                            return self._create_risk_violation(rule, metric, current_value, min_limit, "BELOW")
                    
                    elif isinstance(limit, (int, float)) and current_value > limit:
                        return self._create_risk_violation(rule, metric, current_value, limit, "EXCEEDED")
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
            return None
    
    def _create_risk_violation(self, rule: ComplianceRule, metric: str, current_value: float, 
                              limit_value: float, violation_type: str) -> ComplianceViolation:
        """Create risk violation record"""
        return ComplianceViolation(
            violation_id=str(uuid.uuid4()),
            rule_id=rule.rule_id,
            entity_id=self.entity.entity_id,
            violation_type=f"RISK_LIMIT_{violation_type}",
            description=f"Risk metric {metric} ({current_value:.4f}) {violation_type.lower()} limit ({limit_value:.4f})",
            severity=rule.severity,
            detected_time=datetime.now(timezone.utc),
            regulatory_impact="Risk management review required"
        )
    
    def _check_reporting_requirements(self, rule: ComplianceRule, trading_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check reporting requirement compliance"""
        try:
            # Check if required reports have been generated and submitted
            required_reports = rule.parameters.get('required_reports', [])
            reporting_frequency = rule.parameters.get('frequency', 'DAILY')
            
            for report_type in required_reports:
                # Check if report exists for current period
                if not self._report_exists_for_period(report_type, reporting_frequency):
                    return ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        entity_id=self.entity.entity_id,
                        violation_type="MISSING_REPORT",
                        description=f"Required {report_type} report missing for {reporting_frequency} period",
                        severity=rule.severity,
                        detected_time=datetime.now(timezone.utc),
                        regulatory_impact="Regulatory reporting violation"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking reporting requirements: {str(e)}")
            return None
    
    def _check_clearing_obligations(self, rule: ComplianceRule, trading_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check clearing obligation compliance"""
        try:
            transactions = trading_data.get('transactions', [])
            threshold = rule.parameters.get('clearing_threshold', 3000000000)
            
            # Check if transactions meet clearing obligations
            for transaction in transactions:
                if transaction.get('asset_class') in rule.parameters.get('asset_classes', []):
                    notional = transaction.get('notional_value', 0)
                    
                    if notional > threshold and not transaction.get('centrally_cleared', False):
                        return ComplianceViolation(
                            violation_id=str(uuid.uuid4()),
                            rule_id=rule.rule_id,
                            entity_id=self.entity.entity_id,
                            violation_type="CLEARING_OBLIGATION_BREACH",
                            description=f"Transaction {transaction.get('id')} requires central clearing",
                            severity=rule.severity,
                            detected_time=datetime.now(timezone.utc),
                            financial_impact=notional,
                            regulatory_impact="Central clearing violation"
                        )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking clearing obligations: {str(e)}")
            return None
    
    def _check_proprietary_trading(self, rule: ComplianceRule, trading_data: Dict[str, Any]) -> Optional[ComplianceViolation]:
        """Check proprietary trading restrictions (Volcker Rule)"""
        try:
            transactions = trading_data.get('transactions', [])
            permitted_activities = rule.parameters.get('permitted_activities', [])
            
            for transaction in transactions:
                trading_purpose = transaction.get('trading_purpose', 'proprietary')
                
                if trading_purpose not in permitted_activities:
                    return ComplianceViolation(
                        violation_id=str(uuid.uuid4()),
                        rule_id=rule.rule_id,
                        entity_id=self.entity.entity_id,
                        violation_type="PROPRIETARY_TRADING_VIOLATION",
                        description=f"Transaction {transaction.get('id')} may violate proprietary trading restrictions",
                        severity=rule.severity,
                        detected_time=datetime.now(timezone.utc),
                        financial_impact=transaction.get('notional_value', 0),
                        regulatory_impact="Volcker Rule violation"
                    )
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking proprietary trading: {str(e)}")
            return None
    
    def _report_exists_for_period(self, report_type: str, frequency: str) -> bool:
        """Check if required report exists for the reporting period"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Determine reporting period based on frequency
            if frequency == "DAILY":
                period_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            elif frequency == "WEEKLY":
                days_since_monday = current_time.weekday()
                period_start = current_time - timedelta(days=days_since_monday)
                period_start = period_start.replace(hour=0, minute=0, second=0, microsecond=0)
            elif frequency == "MONTHLY":
                period_start = current_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            else:
                return True  # Assume compliant for other frequencies
            
            # Check if report exists for this period
            for report in self.reports.values():
                if (report.report_type == report_type and 
                    report.reporting_period[0] >= period_start and
                    report.submission_status in ["SUBMITTED", "ACKNOWLEDGED"]):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking report existence: {str(e)}")
            return True  # Assume compliant on error
    
    def resolve_violation(self, violation_id: str, resolution_notes: str) -> bool:
        """
        Resolve a compliance violation
        
        Args:
            violation_id: ID of violation to resolve
            resolution_notes: Notes on how violation was resolved
            
        Returns:
            True if violation resolved successfully
        """
        try:
            for violation in self.violations:
                if violation.violation_id == violation_id:
                    violation.resolved_time = datetime.now(timezone.utc)
                    violation.resolution_notes = resolution_notes
                    violation.status = ComplianceStatus.COMPLIANT
                    
                    logger.info(f"Resolved violation {violation_id}")
                    return True
            
            logger.warning(f"Violation {violation_id} not found")
            return False
            
        except Exception as e:
            logger.error(f"Error resolving violation: {str(e)}")
            return False
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get current compliance status summary"""
        try:
            active_violations = [v for v in self.violations if v.resolved_time is None]
            
            violations_by_severity = {}
            for violation in active_violations:
                severity = violation.severity
                violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
            
            violations_by_regime = {}
            for violation in active_violations:
                # Get rule to determine regime
                rule = self.compliance_rules.get(violation.rule_id)
                if rule:
                    regime = rule.regulatory_regime.value
                    violations_by_regime[regime] = violations_by_regime.get(regime, 0) + 1
            
            return {
                'entity_id': self.entity.entity_id,
                'total_active_violations': len(active_violations),
                'violations_by_severity': violations_by_severity,
                'violations_by_regime': violations_by_regime,
                'total_rules': len(self.compliance_rules),
                'active_rules': len([r for r in self.compliance_rules.values() if r.active]),
                'last_check_time': self.last_check_time.isoformat(),
                'compliance_score': max(0, 100 - len(active_violations) * 10)  # Simple scoring
            }
            
        except Exception as e:
            logger.error(f"Error generating compliance summary: {str(e)}")
            return {}
    
    def start_monitoring(self, check_interval: int = 300):
        """
        Start continuous compliance monitoring
        
        Args:
            check_interval: Monitoring interval in seconds
        """
        try:
            self.monitoring_active = True
            logger.info(f"Started compliance monitoring (interval: {check_interval}s)")
            
        except Exception as e:
            logger.error(f"Error starting monitoring: {str(e)}")
    
    def stop_monitoring(self):
        """Stop compliance monitoring"""
        try:
            self.monitoring_active = False
            logger.info("Stopped compliance monitoring")
            
        except Exception as e:
            logger.error(f"Error stopping monitoring: {str(e)}")

# Export key classes
__all__ = [
    'RegulatoryFramework',
    'RegulatoryEntity', 
    'ComplianceRule',
    'ComplianceViolation',
    'RegulatoryReport',
    'RegulatoryRegime',
    'ComplianceStatus',
    'ReportingFrequency'
]