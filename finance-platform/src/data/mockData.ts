import type {
  Transaction,
  ReconciliationCase,
  LedgerEntry,
  Alert,
  BankBalance,
  PSPBalance,
} from '../types';

export const transactions: Transaction[] = [
  { id: 'TXN-001', type: 'deposit', amount: 25000, currency: 'USD', status: 'completed', sourceSystem: 'PSP-Alpha', client: 'Acme Corp', psp: 'Stripe', bank: 'Chase', timestamp: '2026-05-18T08:23:00Z', linkedRecords: ['LED-001', 'LED-002'], description: 'Client deposit via wire' },
  { id: 'TXN-002', type: 'withdrawal', amount: 12500, currency: 'USD', status: 'completed', sourceSystem: 'PSP-Beta', client: 'Globe Ltd', psp: 'Adyen', bank: 'HSBC', timestamp: '2026-05-18T09:15:00Z', linkedRecords: ['LED-003', 'LED-004'], description: 'Client withdrawal request' },
  { id: 'TXN-003', type: 'deposit', amount: 8700, currency: 'EUR', status: 'pending', sourceSystem: 'PSP-Alpha', client: 'NovaTech', psp: 'Stripe', bank: 'Deutsche', timestamp: '2026-05-18T09:42:00Z', linkedRecords: ['LED-005'], description: 'Pending deposit confirmation' },
  { id: 'TXN-004', type: 'transfer', amount: 50000, currency: 'USD', status: 'completed', sourceSystem: 'Internal', client: 'Internal', psp: 'N/A', bank: 'Chase', timestamp: '2026-05-18T10:01:00Z', linkedRecords: ['LED-006', 'LED-007'], description: 'Treasury transfer' },
  { id: 'TXN-005', type: 'fee', amount: 125, currency: 'USD', status: 'completed', sourceSystem: 'PSP-Alpha', client: 'Acme Corp', psp: 'Stripe', bank: 'Chase', timestamp: '2026-05-18T08:23:05Z', linkedRecords: ['LED-008'], description: 'Processing fee' },
  { id: 'TXN-006', type: 'commission', amount: 375, currency: 'USD', status: 'completed', sourceSystem: 'IB-System', client: 'IB-Partner-01', psp: 'N/A', bank: 'Chase', timestamp: '2026-05-18T10:30:00Z', linkedRecords: ['LED-009'], description: 'IB commission payout' },
  { id: 'TXN-007', type: 'deposit', amount: 150000, currency: 'USD', status: 'completed', sourceSystem: 'PSP-Gamma', client: 'MegaFund', psp: 'Worldpay', bank: 'Barclays', timestamp: '2026-05-17T14:22:00Z', linkedRecords: ['LED-010', 'LED-011'], description: 'Large client deposit' },
  { id: 'TXN-008', type: 'withdrawal', amount: 45000, currency: 'GBP', status: 'failed', sourceSystem: 'PSP-Beta', client: 'BritCo', psp: 'Adyen', bank: 'Barclays', timestamp: '2026-05-17T16:45:00Z', linkedRecords: [], description: 'Failed withdrawal - insufficient PSP balance' },
  { id: 'TXN-009', type: 'deposit', amount: 32000, currency: 'USD', status: 'completed', sourceSystem: 'PSP-Alpha', client: 'SolarInc', psp: 'Stripe', bank: 'Chase', timestamp: '2026-05-17T11:10:00Z', linkedRecords: ['LED-012', 'LED-013'], description: 'Client deposit' },
  { id: 'TXN-010', type: 'withdrawal', amount: 18200, currency: 'EUR', status: 'pending', sourceSystem: 'PSP-Beta', client: 'EuroTrade', psp: 'Adyen', bank: 'Deutsche', timestamp: '2026-05-18T07:50:00Z', linkedRecords: ['LED-014'], description: 'Pending withdrawal' },
  { id: 'TXN-011', type: 'fee', amount: 89, currency: 'USD', status: 'completed', sourceSystem: 'PSP-Gamma', client: 'MegaFund', psp: 'Worldpay', bank: 'Barclays', timestamp: '2026-05-17T14:22:05Z', linkedRecords: ['LED-015'], description: 'Processing fee' },
  { id: 'TXN-012', type: 'deposit', amount: 5600, currency: 'USD', status: 'reversed', sourceSystem: 'PSP-Alpha', client: 'QuickPay', psp: 'Stripe', bank: 'Chase', timestamp: '2026-05-16T09:30:00Z', linkedRecords: ['LED-016', 'LED-017'], description: 'Reversed deposit - chargeback' },
  { id: 'TXN-013', type: 'transfer', amount: 200000, currency: 'USD', status: 'completed', sourceSystem: 'Internal', client: 'Internal', psp: 'N/A', bank: 'Chase', timestamp: '2026-05-16T15:00:00Z', linkedRecords: ['LED-018', 'LED-019'], description: 'Liquidity rebalancing' },
  { id: 'TXN-014', type: 'commission', amount: 1250, currency: 'USD', status: 'completed', sourceSystem: 'IB-System', client: 'IB-Partner-02', psp: 'N/A', bank: 'Chase', timestamp: '2026-05-16T16:00:00Z', linkedRecords: ['LED-020'], description: 'Monthly IB commission' },
  { id: 'TXN-015', type: 'deposit', amount: 72000, currency: 'USD', status: 'completed', sourceSystem: 'PSP-Alpha', client: 'TradeCo', psp: 'Stripe', bank: 'Chase', timestamp: '2026-05-15T10:20:00Z', linkedRecords: ['LED-021', 'LED-022'], description: 'Client deposit' },
];

export const reconciliationCases: ReconciliationCase[] = [
  { id: 'REC-001', transactionId: 'TXN-001', status: 'matched', caseStatus: 'closed', amount: 25000, expectedAmount: 25000, source: 'PSP-Alpha', target: 'Bank-Chase', matchScore: 100, createdAt: '2026-05-18T08:25:00Z', updatedAt: '2026-05-18T08:26:00Z', notes: 'Auto-matched' },
  { id: 'REC-002', transactionId: 'TXN-003', status: 'partial', caseStatus: 'investigating', amount: 8700, expectedAmount: 9000, source: 'PSP-Alpha', target: 'Bank-Deutsche', matchScore: 72, createdAt: '2026-05-18T09:45:00Z', updatedAt: '2026-05-18T10:00:00Z', notes: 'Amount mismatch — FX differential suspected' },
  { id: 'REC-003', transactionId: 'TXN-008', status: 'exception', caseStatus: 'open', amount: 45000, expectedAmount: 45000, source: 'PSP-Beta', target: 'Bank-Barclays', matchScore: 0, createdAt: '2026-05-17T17:00:00Z', updatedAt: '2026-05-17T17:00:00Z', notes: 'Failed withdrawal — no bank record found' },
  { id: 'REC-004', transactionId: 'TXN-007', status: 'matched', caseStatus: 'closed', amount: 150000, expectedAmount: 150000, source: 'PSP-Gamma', target: 'Bank-Barclays', matchScore: 100, createdAt: '2026-05-17T14:30:00Z', updatedAt: '2026-05-17T14:31:00Z', notes: 'Auto-matched' },
  { id: 'REC-005', transactionId: 'TXN-012', status: 'exception', caseStatus: 'investigating', amount: 5600, expectedAmount: 5600, source: 'PSP-Alpha', target: 'Bank-Chase', matchScore: 50, createdAt: '2026-05-16T10:00:00Z', updatedAt: '2026-05-16T15:00:00Z', notes: 'Chargeback reversal — investigating' },
  { id: 'REC-006', transactionId: 'TXN-009', status: 'matched', caseStatus: 'closed', amount: 32000, expectedAmount: 32000, source: 'PSP-Alpha', target: 'Bank-Chase', matchScore: 100, createdAt: '2026-05-17T11:15:00Z', updatedAt: '2026-05-17T11:16:00Z', notes: 'Auto-matched' },
  { id: 'REC-007', transactionId: 'TXN-010', status: 'partial', caseStatus: 'open', amount: 18200, expectedAmount: 18500, source: 'PSP-Beta', target: 'Bank-Deutsche', matchScore: 65, createdAt: '2026-05-18T08:00:00Z', updatedAt: '2026-05-18T08:00:00Z', notes: 'Pending — amount discrepancy' },
  { id: 'REC-008', transactionId: 'TXN-002', status: 'matched', caseStatus: 'closed', amount: 12500, expectedAmount: 12500, source: 'PSP-Beta', target: 'Bank-HSBC', matchScore: 100, createdAt: '2026-05-18T09:20:00Z', updatedAt: '2026-05-18T09:21:00Z', notes: 'Auto-matched' },
];

export const ledgerEntries: LedgerEntry[] = [
  { id: 'LED-001', type: 'debit', account: 'Company Cash', accountCategory: 'company_cash', amount: 25000, currency: 'USD', reference: 'TXN-001', timestamp: '2026-05-18T08:23:00Z', description: 'Cash received from deposit' },
  { id: 'LED-002', type: 'credit', account: 'Client Liability - Acme Corp', accountCategory: 'client_liability', amount: 25000, currency: 'USD', reference: 'TXN-001', timestamp: '2026-05-18T08:23:00Z', description: 'Client balance credited' },
  { id: 'LED-003', type: 'credit', account: 'Company Cash', accountCategory: 'company_cash', amount: 12500, currency: 'USD', reference: 'TXN-002', timestamp: '2026-05-18T09:15:00Z', description: 'Cash disbursed for withdrawal' },
  { id: 'LED-004', type: 'debit', account: 'Client Liability - Globe Ltd', accountCategory: 'client_liability', amount: 12500, currency: 'USD', reference: 'TXN-002', timestamp: '2026-05-18T09:15:00Z', description: 'Client balance debited' },
  { id: 'LED-005', type: 'debit', account: 'PSP Receivable - Stripe', accountCategory: 'psp_accounts', amount: 8700, currency: 'EUR', reference: 'TXN-003', timestamp: '2026-05-18T09:42:00Z', description: 'Pending PSP settlement' },
  { id: 'LED-006', type: 'debit', account: 'Company Cash - Operating', accountCategory: 'company_cash', amount: 50000, currency: 'USD', reference: 'TXN-004', timestamp: '2026-05-18T10:01:00Z', description: 'Treasury transfer in' },
  { id: 'LED-007', type: 'credit', account: 'Company Cash - Reserve', accountCategory: 'company_cash', amount: 50000, currency: 'USD', reference: 'TXN-004', timestamp: '2026-05-18T10:01:00Z', description: 'Treasury transfer out' },
  { id: 'LED-008', type: 'debit', account: 'Fee Income', accountCategory: 'fee_accounts', amount: 125, currency: 'USD', reference: 'TXN-005', timestamp: '2026-05-18T08:23:05Z', description: 'Processing fee collected' },
  { id: 'LED-009', type: 'credit', account: 'Commission Payable - IB-01', accountCategory: 'commission_accounts', amount: 375, currency: 'USD', reference: 'TXN-006', timestamp: '2026-05-18T10:30:00Z', description: 'IB commission accrued' },
  { id: 'LED-010', type: 'debit', account: 'Company Cash', accountCategory: 'company_cash', amount: 150000, currency: 'USD', reference: 'TXN-007', timestamp: '2026-05-17T14:22:00Z', description: 'Large deposit received' },
  { id: 'LED-011', type: 'credit', account: 'Client Liability - MegaFund', accountCategory: 'client_liability', amount: 150000, currency: 'USD', reference: 'TXN-007', timestamp: '2026-05-17T14:22:00Z', description: 'Client balance credited' },
  { id: 'LED-012', type: 'debit', account: 'Company Cash', accountCategory: 'company_cash', amount: 32000, currency: 'USD', reference: 'TXN-009', timestamp: '2026-05-17T11:10:00Z', description: 'Deposit received' },
  { id: 'LED-013', type: 'credit', account: 'Client Liability - SolarInc', accountCategory: 'client_liability', amount: 32000, currency: 'USD', reference: 'TXN-009', timestamp: '2026-05-17T11:10:00Z', description: 'Client balance credited' },
  { id: 'LED-014', type: 'credit', account: 'Client Liability - EuroTrade', accountCategory: 'client_liability', amount: 18200, currency: 'EUR', reference: 'TXN-010', timestamp: '2026-05-18T07:50:00Z', description: 'Pending withdrawal reserved' },
  { id: 'LED-015', type: 'debit', account: 'Fee Income', accountCategory: 'fee_accounts', amount: 89, currency: 'USD', reference: 'TXN-011', timestamp: '2026-05-17T14:22:05Z', description: 'Processing fee collected' },
];

export const alerts: Alert[] = [
  { id: 'ALT-001', title: 'Cash Imbalance Detected', category: 'financial', severity: 'critical', status: 'open', description: 'Company cash account shows $12,400 discrepancy between expected and actual balance.', linkedTransactions: ['TXN-004', 'TXN-013'], createdAt: '2026-05-18T10:15:00Z' },
  { id: 'ALT-002', title: 'Withdrawal Spike Alert', category: 'financial', severity: 'high', status: 'investigating', description: 'Withdrawal volume increased 340% compared to 7-day average. Monitoring for liquidity impact.', linkedTransactions: ['TXN-002', 'TXN-008', 'TXN-010'], createdAt: '2026-05-18T09:30:00Z' },
  { id: 'ALT-003', title: 'PSP Settlement Delay - Adyen', category: 'operational', severity: 'medium', status: 'open', description: 'Adyen settlement batch delayed by 4 hours. Estimated funds: €63,200.', linkedTransactions: ['TXN-002', 'TXN-010'], createdAt: '2026-05-18T08:00:00Z' },
  { id: 'ALT-004', title: 'Reconciliation Exception Rate High', category: 'operational', severity: 'high', status: 'investigating', description: 'Exception rate at 18% for today, above 5% threshold. 3 unresolved cases.', linkedTransactions: ['TXN-008', 'TXN-012'], createdAt: '2026-05-18T10:00:00Z' },
  { id: 'ALT-005', title: 'Chargeback Reversal', category: 'financial', severity: 'medium', status: 'resolved', description: 'Chargeback on TXN-012 processed. Client QuickPay debited $5,600.', linkedTransactions: ['TXN-012'], createdAt: '2026-05-16T10:30:00Z', resolvedAt: '2026-05-17T09:00:00Z' },
  { id: 'ALT-006', title: 'Low Liquidity Buffer Warning', category: 'financial', severity: 'critical', status: 'open', description: 'Available liquidity buffer dropped below 15% threshold. Current: 11.2%.', linkedTransactions: [], createdAt: '2026-05-18T07:00:00Z' },
];

export const bankBalances: BankBalance[] = [
  { id: 'BB-001', bank: 'Chase', currency: 'USD', balance: 2450000, lastUpdated: '2026-05-18T10:00:00Z' },
  { id: 'BB-002', bank: 'HSBC', currency: 'USD', balance: 890000, lastUpdated: '2026-05-18T09:30:00Z' },
  { id: 'BB-003', bank: 'Deutsche Bank', currency: 'EUR', balance: 1250000, lastUpdated: '2026-05-18T09:00:00Z' },
  { id: 'BB-004', bank: 'Barclays', currency: 'GBP', balance: 675000, lastUpdated: '2026-05-18T08:45:00Z' },
  { id: 'BB-005', bank: 'Barclays', currency: 'USD', balance: 320000, lastUpdated: '2026-05-18T08:45:00Z' },
];

export const pspBalances: PSPBalance[] = [
  { id: 'PSP-001', psp: 'Stripe', currency: 'USD', balance: 185000, pendingIn: 32000, pendingOut: 8500, lastUpdated: '2026-05-18T10:00:00Z' },
  { id: 'PSP-002', psp: 'Stripe', currency: 'EUR', balance: 45000, pendingIn: 8700, pendingOut: 0, lastUpdated: '2026-05-18T10:00:00Z' },
  { id: 'PSP-003', psp: 'Adyen', currency: 'USD', balance: 92000, pendingIn: 0, pendingOut: 12500, lastUpdated: '2026-05-18T09:00:00Z' },
  { id: 'PSP-004', psp: 'Adyen', currency: 'EUR', balance: 63200, pendingIn: 0, pendingOut: 18200, lastUpdated: '2026-05-18T09:00:00Z' },
  { id: 'PSP-005', psp: 'Worldpay', currency: 'USD', balance: 210000, pendingIn: 0, pendingOut: 0, lastUpdated: '2026-05-18T08:00:00Z' },
  { id: 'PSP-006', psp: 'Worldpay', currency: 'GBP', balance: 55000, pendingIn: 0, pendingOut: 45000, lastUpdated: '2026-05-18T08:00:00Z' },
];

export const cashFlowData = [
  { date: 'May 12', deposits: 185000, withdrawals: 72000, netFlow: 113000 },
  { date: 'May 13', deposits: 210000, withdrawals: 95000, netFlow: 115000 },
  { date: 'May 14', deposits: 145000, withdrawals: 120000, netFlow: 25000 },
  { date: 'May 15', deposits: 290000, withdrawals: 88000, netFlow: 202000 },
  { date: 'May 16', deposits: 178000, withdrawals: 156000, netFlow: 22000 },
  { date: 'May 17', deposits: 320000, withdrawals: 105000, netFlow: 215000 },
  { date: 'May 18', deposits: 265700, withdrawals: 75700, netFlow: 190000 },
];

export const liquidityTrend = [
  { date: 'May 12', available: 4200000, buffer: 18.2 },
  { date: 'May 13', available: 4315000, buffer: 17.8 },
  { date: 'May 14', available: 4050000, buffer: 15.5 },
  { date: 'May 15', available: 4400000, buffer: 16.1 },
  { date: 'May 16', available: 4150000, buffer: 14.2 },
  { date: 'May 17', available: 4580000, buffer: 13.8 },
  { date: 'May 18', available: 4585000, buffer: 11.2 },
];

export const profitabilityData = [
  { name: 'Acme Corp', revenue: 45000, costs: 12000, profit: 33000 },
  { name: 'Globe Ltd', revenue: 28000, costs: 9500, profit: 18500 },
  { name: 'MegaFund', revenue: 82000, costs: 18000, profit: 64000 },
  { name: 'NovaTech', revenue: 15000, costs: 6200, profit: 8800 },
  { name: 'SolarInc', revenue: 32000, costs: 8800, profit: 23200 },
  { name: 'TradeCo', revenue: 51000, costs: 14500, profit: 36500 },
];

export const ibProfitabilityData = [
  { name: 'IB-Partner-01', clients: 42, volume: 1250000, commission: 18750, netRevenue: 62500 },
  { name: 'IB-Partner-02', clients: 28, volume: 890000, commission: 13350, netRevenue: 44500 },
  { name: 'IB-Partner-03', clients: 65, volume: 2100000, commission: 31500, netRevenue: 105000 },
  { name: 'IB-Partner-04', clients: 15, volume: 420000, commission: 6300, netRevenue: 21000 },
];

export const monthlyKPIs = [
  { month: 'Jan', netFlow: 1200000, opCosts: 85000, exceptions: 12 },
  { month: 'Feb', netFlow: 980000, opCosts: 78000, exceptions: 8 },
  { month: 'Mar', netFlow: 1450000, opCosts: 92000, exceptions: 15 },
  { month: 'Apr', netFlow: 1100000, opCosts: 88000, exceptions: 10 },
  { month: 'May', netFlow: 1680000, opCosts: 95000, exceptions: 18 },
];
