// ─── Common ───
export type Status = 'active' | 'inactive' | 'pending' | 'completed' | 'cancelled' | 'draft';
export type Priority = 'low' | 'medium' | 'high' | 'critical';

// ─── HR ───
export interface Employee {
  id: string;
  name: string;
  email: string;
  phone: string;
  department: string;
  position: string;
  status: 'active' | 'on-leave' | 'terminated';
  hireDate: string;
  salary: number;
  avatar?: string;
  manager?: string;
}

export interface Department {
  id: string;
  name: string;
  head: string;
  employeeCount: number;
  budget: number;
}

export interface PayrollRecord {
  id: string;
  employeeId: string;
  employeeName: string;
  department: string;
  period: string;
  baseSalary: number;
  bonus: number;
  deductions: number;
  netPay: number;
  status: 'processed' | 'pending' | 'failed';
  paidDate?: string;
}

export interface LeaveRequest {
  id: string;
  employeeId: string;
  employeeName: string;
  type: 'vacation' | 'sick' | 'personal' | 'maternity';
  startDate: string;
  endDate: string;
  status: 'approved' | 'pending' | 'rejected';
  reason: string;
}

// ─── Inventory ───
export interface Product {
  id: string;
  sku: string;
  name: string;
  category: string;
  price: number;
  cost: number;
  stock: number;
  reorderLevel: number;
  warehouse: string;
  status: 'in-stock' | 'low-stock' | 'out-of-stock';
  lastRestocked: string;
}

export interface Warehouse {
  id: string;
  name: string;
  location: string;
  capacity: number;
  utilization: number;
  manager: string;
  productCount: number;
}

export interface StockMovement {
  id: string;
  productId: string;
  productName: string;
  type: 'in' | 'out' | 'transfer' | 'adjustment';
  quantity: number;
  fromWarehouse?: string;
  toWarehouse?: string;
  date: string;
  reference: string;
  note: string;
}

// ─── Sales ───
export interface Customer {
  id: string;
  name: string;
  company: string;
  email: string;
  phone: string;
  address: string;
  totalOrders: number;
  totalSpent: number;
  status: 'active' | 'inactive';
  lastOrderDate: string;
  segment: 'enterprise' | 'mid-market' | 'small-business';
}

export interface SalesOrder {
  id: string;
  orderNumber: string;
  customerId: string;
  customerName: string;
  items: OrderItem[];
  subtotal: number;
  tax: number;
  total: number;
  status: 'draft' | 'confirmed' | 'shipped' | 'delivered' | 'cancelled';
  orderDate: string;
  deliveryDate?: string;
  paymentStatus: 'paid' | 'partial' | 'unpaid';
}

export interface OrderItem {
  productId: string;
  productName: string;
  quantity: number;
  unitPrice: number;
  total: number;
}

export interface SalesPipeline {
  id: string;
  dealName: string;
  customer: string;
  value: number;
  stage: 'lead' | 'qualified' | 'proposal' | 'negotiation' | 'closed-won' | 'closed-lost';
  probability: number;
  expectedClose: string;
  assignedTo: string;
}

// ─── Procurement ───
export interface Supplier {
  id: string;
  name: string;
  contactPerson: string;
  email: string;
  phone: string;
  address: string;
  rating: number;
  totalOrders: number;
  status: 'active' | 'inactive' | 'blacklisted';
  paymentTerms: string;
  leadTimeDays: number;
}

export interface PurchaseOrder {
  id: string;
  poNumber: string;
  supplierId: string;
  supplierName: string;
  items: PurchaseItem[];
  subtotal: number;
  tax: number;
  total: number;
  status: 'draft' | 'sent' | 'confirmed' | 'received' | 'cancelled';
  orderDate: string;
  expectedDelivery: string;
  receivedDate?: string;
}

export interface PurchaseItem {
  productId: string;
  productName: string;
  quantity: number;
  unitCost: number;
  total: number;
}

// ─── Accounting ───
export interface Account {
  id: string;
  code: string;
  name: string;
  type: 'asset' | 'liability' | 'equity' | 'revenue' | 'expense';
  balance: number;
  parentAccount?: string;
  isActive: boolean;
}

export interface JournalEntry {
  id: string;
  entryNumber: string;
  date: string;
  description: string;
  lines: JournalLine[];
  status: 'draft' | 'posted' | 'reversed';
  createdBy: string;
}

export interface JournalLine {
  accountId: string;
  accountName: string;
  debit: number;
  credit: number;
  description?: string;
}

export interface Invoice {
  id: string;
  invoiceNumber: string;
  customerId: string;
  customerName: string;
  issueDate: string;
  dueDate: string;
  items: InvoiceItem[];
  subtotal: number;
  tax: number;
  total: number;
  amountPaid: number;
  status: 'draft' | 'sent' | 'paid' | 'overdue' | 'cancelled';
}

export interface InvoiceItem {
  description: string;
  quantity: number;
  rate: number;
  amount: number;
}

// ─── Projects ───
export interface Project {
  id: string;
  name: string;
  description: string;
  client: string;
  manager: string;
  status: 'planning' | 'active' | 'on-hold' | 'completed' | 'cancelled';
  priority: Priority;
  startDate: string;
  endDate: string;
  budget: number;
  spent: number;
  progress: number;
  teamMembers: string[];
}

export interface Task {
  id: string;
  projectId: string;
  title: string;
  description: string;
  assignee: string;
  status: 'todo' | 'in-progress' | 'review' | 'done';
  priority: Priority;
  dueDate: string;
  estimatedHours: number;
  loggedHours: number;
}

// ─── Navigation ───
export interface NavItem {
  label: string;
  path: string;
  icon: string;
  children?: NavItem[];
}
