export type TransactionType = 'deposit' | 'withdrawal' | 'transfer' | 'fee' | 'commission';
export type TransactionStatus = 'completed' | 'pending' | 'failed' | 'reversed';
export type ReconciliationStatus = 'matched' | 'partial' | 'exception';
export type AlertSeverity = 'critical' | 'high' | 'medium' | 'low';
export type AlertCategory = 'financial' | 'operational';
export type CaseStatus = 'open' | 'investigating' | 'resolved' | 'closed';
export type LedgerEntryType = 'debit' | 'credit';

export interface Transaction {
  id: string;
  type: TransactionType;
  amount: number;
  currency: string;
  status: TransactionStatus;
  sourceSystem: string;
  client: string;
  psp: string;
  bank: string;
  timestamp: string;
  linkedRecords: string[];
  description: string;
}

export interface ReconciliationCase {
  id: string;
  transactionId: string;
  status: ReconciliationStatus;
  caseStatus: CaseStatus;
  amount: number;
  expectedAmount: number;
  source: string;
  target: string;
  matchScore: number;
  createdAt: string;
  updatedAt: string;
  notes: string;
}

export interface LedgerEntry {
  id: string;
  type: LedgerEntryType;
  account: string;
  accountCategory: string;
  amount: number;
  currency: string;
  reference: string;
  timestamp: string;
  description: string;
}

export interface Alert {
  id: string;
  title: string;
  category: AlertCategory;
  severity: AlertSeverity;
  status: CaseStatus;
  description: string;
  linkedTransactions: string[];
  createdAt: string;
  resolvedAt?: string;
}

export interface BankBalance {
  id: string;
  bank: string;
  currency: string;
  balance: number;
  lastUpdated: string;
}

export interface PSPBalance {
  id: string;
  psp: string;
  currency: string;
  balance: number;
  pendingIn: number;
  pendingOut: number;
  lastUpdated: string;
}

export interface KPI {
  label: string;
  value: string | number;
  change: number;
  trend: 'up' | 'down' | 'flat';
  prefix?: string;
}
