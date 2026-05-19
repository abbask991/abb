import type {
  Employee, Department, PayrollRecord, LeaveRequest,
  Product, Warehouse, StockMovement,
  Customer, SalesOrder, SalesPipeline,
  Supplier, PurchaseOrder,
  Account, JournalEntry, Invoice,
  Project, Task,
} from '../types';

// ─── HR ───

export const departments: Department[] = [
  { id: 'D001', name: 'Engineering', head: 'Sarah Chen', employeeCount: 42, budget: 2800000 },
  { id: 'D002', name: 'Sales', head: 'James Miller', employeeCount: 28, budget: 1500000 },
  { id: 'D003', name: 'Marketing', head: 'Emily Davis', employeeCount: 15, budget: 900000 },
  { id: 'D004', name: 'Human Resources', head: 'Maria Garcia', employeeCount: 8, budget: 450000 },
  { id: 'D005', name: 'Finance', head: 'Robert Kim', employeeCount: 12, budget: 650000 },
  { id: 'D006', name: 'Operations', head: 'David Wilson', employeeCount: 20, budget: 1100000 },
  { id: 'D007', name: 'Customer Support', head: 'Lisa Brown', employeeCount: 18, budget: 750000 },
  { id: 'D008', name: 'Product', head: 'Alex Turner', employeeCount: 10, budget: 600000 },
];

export const employees: Employee[] = [
  { id: 'E001', name: 'Sarah Chen', email: 'sarah.chen@company.com', phone: '+1-555-0101', department: 'Engineering', position: 'VP of Engineering', status: 'active', hireDate: '2019-03-15', salary: 185000, manager: 'CEO' },
  { id: 'E002', name: 'James Miller', email: 'james.miller@company.com', phone: '+1-555-0102', department: 'Sales', position: 'VP of Sales', status: 'active', hireDate: '2019-06-01', salary: 170000, manager: 'CEO' },
  { id: 'E003', name: 'Emily Davis', email: 'emily.davis@company.com', phone: '+1-555-0103', department: 'Marketing', position: 'Marketing Director', status: 'active', hireDate: '2020-01-10', salary: 145000, manager: 'CEO' },
  { id: 'E004', name: 'Maria Garcia', email: 'maria.garcia@company.com', phone: '+1-555-0104', department: 'Human Resources', position: 'HR Director', status: 'active', hireDate: '2019-08-20', salary: 130000, manager: 'CEO' },
  { id: 'E005', name: 'Robert Kim', email: 'robert.kim@company.com', phone: '+1-555-0105', department: 'Finance', position: 'CFO', status: 'active', hireDate: '2018-11-05', salary: 195000, manager: 'CEO' },
  { id: 'E006', name: 'John Park', email: 'john.park@company.com', phone: '+1-555-0106', department: 'Engineering', position: 'Senior Developer', status: 'active', hireDate: '2020-04-12', salary: 135000, manager: 'Sarah Chen' },
  { id: 'E007', name: 'Lisa Brown', email: 'lisa.brown@company.com', phone: '+1-555-0107', department: 'Customer Support', position: 'Support Manager', status: 'active', hireDate: '2020-07-18', salary: 95000, manager: 'CEO' },
  { id: 'E008', name: 'David Wilson', email: 'david.wilson@company.com', phone: '+1-555-0108', department: 'Operations', position: 'COO', status: 'active', hireDate: '2019-02-01', salary: 190000, manager: 'CEO' },
  { id: 'E009', name: 'Alex Turner', email: 'alex.turner@company.com', phone: '+1-555-0109', department: 'Product', position: 'Head of Product', status: 'active', hireDate: '2020-09-01', salary: 155000, manager: 'CEO' },
  { id: 'E010', name: 'Rachel Lee', email: 'rachel.lee@company.com', phone: '+1-555-0110', department: 'Engineering', position: 'Frontend Lead', status: 'active', hireDate: '2021-01-15', salary: 125000, manager: 'Sarah Chen' },
  { id: 'E011', name: 'Michael Torres', email: 'michael.torres@company.com', phone: '+1-555-0111', department: 'Sales', position: 'Account Executive', status: 'active', hireDate: '2021-03-20', salary: 85000, manager: 'James Miller' },
  { id: 'E012', name: 'Sophia Nguyen', email: 'sophia.nguyen@company.com', phone: '+1-555-0112', department: 'Marketing', position: 'Content Strategist', status: 'on-leave', hireDate: '2021-06-10', salary: 78000, manager: 'Emily Davis' },
  { id: 'E013', name: 'Daniel Johnson', email: 'daniel.johnson@company.com', phone: '+1-555-0113', department: 'Engineering', position: 'Backend Developer', status: 'active', hireDate: '2021-08-05', salary: 115000, manager: 'Sarah Chen' },
  { id: 'E014', name: 'Amanda White', email: 'amanda.white@company.com', phone: '+1-555-0114', department: 'Finance', position: 'Senior Accountant', status: 'active', hireDate: '2020-11-15', salary: 95000, manager: 'Robert Kim' },
  { id: 'E015', name: 'Chris Anderson', email: 'chris.anderson@company.com', phone: '+1-555-0115', department: 'Operations', position: 'Logistics Manager', status: 'active', hireDate: '2021-02-28', salary: 88000, manager: 'David Wilson' },
  { id: 'E016', name: 'Jessica Taylor', email: 'jessica.taylor@company.com', phone: '+1-555-0116', department: 'Sales', position: 'Sales Manager', status: 'active', hireDate: '2020-05-15', salary: 110000, manager: 'James Miller' },
  { id: 'E017', name: 'Brian Harris', email: 'brian.harris@company.com', phone: '+1-555-0117', department: 'Engineering', position: 'DevOps Engineer', status: 'active', hireDate: '2021-10-01', salary: 120000, manager: 'Sarah Chen' },
  { id: 'E018', name: 'Olivia Martinez', email: 'olivia.martinez@company.com', phone: '+1-555-0118', department: 'Human Resources', position: 'Recruiter', status: 'active', hireDate: '2022-01-10', salary: 72000, manager: 'Maria Garcia' },
  { id: 'E019', name: 'Kevin Wright', email: 'kevin.wright@company.com', phone: '+1-555-0119', department: 'Customer Support', position: 'Support Agent', status: 'terminated', hireDate: '2021-04-15', salary: 55000, manager: 'Lisa Brown' },
  { id: 'E020', name: 'Nina Patel', email: 'nina.patel@company.com', phone: '+1-555-0120', department: 'Product', position: 'Product Manager', status: 'active', hireDate: '2022-03-01', salary: 120000, manager: 'Alex Turner' },
];

export const payrollRecords: PayrollRecord[] = [
  { id: 'PR001', employeeId: 'E001', employeeName: 'Sarah Chen', department: 'Engineering', period: '2026-04', baseSalary: 15416.67, bonus: 2000, deductions: 3850, netPay: 13566.67, status: 'processed', paidDate: '2026-04-30' },
  { id: 'PR002', employeeId: 'E002', employeeName: 'James Miller', department: 'Sales', period: '2026-04', baseSalary: 14166.67, bonus: 5200, deductions: 3540, netPay: 15826.67, status: 'processed', paidDate: '2026-04-30' },
  { id: 'PR003', employeeId: 'E005', employeeName: 'Robert Kim', department: 'Finance', period: '2026-04', baseSalary: 16250.00, bonus: 1500, deductions: 4100, netPay: 13650.00, status: 'processed', paidDate: '2026-04-30' },
  { id: 'PR004', employeeId: 'E006', employeeName: 'John Park', department: 'Engineering', period: '2026-05', baseSalary: 11250.00, bonus: 800, deductions: 2800, netPay: 9250.00, status: 'pending' },
  { id: 'PR005', employeeId: 'E010', employeeName: 'Rachel Lee', department: 'Engineering', period: '2026-05', baseSalary: 10416.67, bonus: 600, deductions: 2600, netPay: 8416.67, status: 'pending' },
  { id: 'PR006', employeeId: 'E011', employeeName: 'Michael Torres', department: 'Sales', period: '2026-05', baseSalary: 7083.33, bonus: 3200, deductions: 1800, netPay: 8483.33, status: 'pending' },
];

export const leaveRequests: LeaveRequest[] = [
  { id: 'LR001', employeeId: 'E012', employeeName: 'Sophia Nguyen', type: 'maternity', startDate: '2026-04-01', endDate: '2026-07-01', status: 'approved', reason: 'Maternity leave' },
  { id: 'LR002', employeeId: 'E006', employeeName: 'John Park', type: 'vacation', startDate: '2026-05-20', endDate: '2026-05-30', status: 'approved', reason: 'Family vacation' },
  { id: 'LR003', employeeId: 'E013', employeeName: 'Daniel Johnson', type: 'sick', startDate: '2026-05-15', endDate: '2026-05-16', status: 'approved', reason: 'Flu' },
  { id: 'LR004', employeeId: 'E016', employeeName: 'Jessica Taylor', type: 'vacation', startDate: '2026-06-01', endDate: '2026-06-10', status: 'pending', reason: 'Travel plans' },
  { id: 'LR005', employeeId: 'E017', employeeName: 'Brian Harris', type: 'personal', startDate: '2026-05-22', endDate: '2026-05-23', status: 'pending', reason: 'Personal matters' },
];

// ─── Inventory ───

export const products: Product[] = [
  { id: 'P001', sku: 'WDG-001', name: 'Industrial Widget A', category: 'Widgets', price: 29.99, cost: 12.50, stock: 2450, reorderLevel: 500, warehouse: 'Main Warehouse', status: 'in-stock', lastRestocked: '2026-05-10' },
  { id: 'P002', sku: 'WDG-002', name: 'Premium Widget B', category: 'Widgets', price: 49.99, cost: 22.00, stock: 180, reorderLevel: 200, warehouse: 'Main Warehouse', status: 'low-stock', lastRestocked: '2026-04-28' },
  { id: 'P003', sku: 'GDG-001', name: 'Gadget Pro X1', category: 'Gadgets', price: 199.99, cost: 85.00, stock: 890, reorderLevel: 100, warehouse: 'East Distribution', status: 'in-stock', lastRestocked: '2026-05-05' },
  { id: 'P004', sku: 'GDG-002', name: 'Gadget Mini S2', category: 'Gadgets', price: 79.99, cost: 35.00, stock: 0, reorderLevel: 150, warehouse: 'Main Warehouse', status: 'out-of-stock', lastRestocked: '2026-03-15' },
  { id: 'P005', sku: 'CMP-001', name: 'Component Alpha', category: 'Components', price: 8.50, cost: 3.20, stock: 12000, reorderLevel: 2000, warehouse: 'Main Warehouse', status: 'in-stock', lastRestocked: '2026-05-12' },
  { id: 'P006', sku: 'CMP-002', name: 'Component Beta', category: 'Components', price: 12.75, cost: 5.80, stock: 8500, reorderLevel: 1500, warehouse: 'West Facility', status: 'in-stock', lastRestocked: '2026-05-08' },
  { id: 'P007', sku: 'ASM-001', name: 'Assembly Kit Pro', category: 'Assemblies', price: 349.99, cost: 150.00, stock: 45, reorderLevel: 50, warehouse: 'East Distribution', status: 'low-stock', lastRestocked: '2026-04-20' },
  { id: 'P008', sku: 'ASM-002', name: 'Assembly Kit Standard', category: 'Assemblies', price: 189.99, cost: 80.00, stock: 320, reorderLevel: 75, warehouse: 'Main Warehouse', status: 'in-stock', lastRestocked: '2026-05-01' },
  { id: 'P009', sku: 'TLS-001', name: 'Precision Tool Set', category: 'Tools', price: 599.99, cost: 280.00, stock: 65, reorderLevel: 20, warehouse: 'West Facility', status: 'in-stock', lastRestocked: '2026-04-15' },
  { id: 'P010', sku: 'TLS-002', name: 'Safety Equipment Pack', category: 'Safety', price: 129.99, cost: 55.00, stock: 410, reorderLevel: 100, warehouse: 'Main Warehouse', status: 'in-stock', lastRestocked: '2026-05-14' },
  { id: 'P011', sku: 'RAW-001', name: 'Steel Rods (Bundle)', category: 'Raw Materials', price: 89.99, cost: 42.00, stock: 1800, reorderLevel: 300, warehouse: 'West Facility', status: 'in-stock', lastRestocked: '2026-05-11' },
  { id: 'P012', sku: 'RAW-002', name: 'Copper Wire (Spool)', category: 'Raw Materials', price: 45.00, cost: 20.00, stock: 95, reorderLevel: 100, warehouse: 'Main Warehouse', status: 'low-stock', lastRestocked: '2026-04-25' },
];

export const warehouses: Warehouse[] = [
  { id: 'W001', name: 'Main Warehouse', location: 'Houston, TX', capacity: 50000, utilization: 78, manager: 'Chris Anderson', productCount: 6 },
  { id: 'W002', name: 'East Distribution', location: 'Atlanta, GA', capacity: 30000, utilization: 62, manager: 'Tom Richards', productCount: 3 },
  { id: 'W003', name: 'West Facility', location: 'Phoenix, AZ', capacity: 35000, utilization: 85, manager: 'Susan Clark', productCount: 3 },
];

export const stockMovements: StockMovement[] = [
  { id: 'SM001', productId: 'P001', productName: 'Industrial Widget A', type: 'in', quantity: 500, toWarehouse: 'Main Warehouse', date: '2026-05-10', reference: 'PO-2026-042', note: 'Regular restock' },
  { id: 'SM002', productId: 'P003', productName: 'Gadget Pro X1', type: 'out', quantity: 120, fromWarehouse: 'East Distribution', date: '2026-05-12', reference: 'SO-2026-089', note: 'Customer order fulfillment' },
  { id: 'SM003', productId: 'P005', productName: 'Component Alpha', type: 'in', quantity: 3000, toWarehouse: 'Main Warehouse', date: '2026-05-12', reference: 'PO-2026-045', note: 'Bulk order receipt' },
  { id: 'SM004', productId: 'P002', productName: 'Premium Widget B', type: 'transfer', quantity: 200, fromWarehouse: 'West Facility', toWarehouse: 'Main Warehouse', date: '2026-05-13', reference: 'TF-2026-012', note: 'Rebalancing stock' },
  { id: 'SM005', productId: 'P007', productName: 'Assembly Kit Pro', type: 'out', quantity: 15, fromWarehouse: 'East Distribution', date: '2026-05-14', reference: 'SO-2026-091', note: 'Enterprise client order' },
  { id: 'SM006', productId: 'P004', productName: 'Gadget Mini S2', type: 'adjustment', quantity: -8, toWarehouse: 'Main Warehouse', date: '2026-05-14', reference: 'ADJ-2026-003', note: 'Damaged inventory write-off' },
];

// ─── Sales ───

export const customers: Customer[] = [
  { id: 'C001', name: 'John Bradley', company: 'Acme Corporation', email: 'john@acme.com', phone: '+1-555-1001', address: '123 Business Ave, New York, NY', totalOrders: 47, totalSpent: 284500, status: 'active', lastOrderDate: '2026-05-14', segment: 'enterprise' },
  { id: 'C002', name: 'Sarah Mitchell', company: 'TechFlow Inc', email: 'sarah@techflow.com', phone: '+1-555-1002', address: '456 Innovation Blvd, San Francisco, CA', totalOrders: 32, totalSpent: 156200, status: 'active', lastOrderDate: '2026-05-10', segment: 'mid-market' },
  { id: 'C003', name: 'Robert Hayes', company: 'Global Dynamics', email: 'robert@globaldyn.com', phone: '+1-555-1003', address: '789 Enterprise Dr, Chicago, IL', totalOrders: 65, totalSpent: 532800, status: 'active', lastOrderDate: '2026-05-18', segment: 'enterprise' },
  { id: 'C004', name: 'Jennifer Walsh', company: 'StartUp Hub', email: 'jennifer@startuphub.com', phone: '+1-555-1004', address: '321 Startup Lane, Austin, TX', totalOrders: 12, totalSpent: 34500, status: 'active', lastOrderDate: '2026-04-28', segment: 'small-business' },
  { id: 'C005', name: 'Michael Chen', company: 'Pacific Trade Co', email: 'michael@pacifictrade.com', phone: '+1-555-1005', address: '555 Harbor Way, Seattle, WA', totalOrders: 28, totalSpent: 198700, status: 'active', lastOrderDate: '2026-05-08', segment: 'mid-market' },
  { id: 'C006', name: 'Laura Bennett', company: 'Summit Industries', email: 'laura@summit.com', phone: '+1-555-1006', address: '100 Mountain Rd, Denver, CO', totalOrders: 41, totalSpent: 312400, status: 'active', lastOrderDate: '2026-05-16', segment: 'enterprise' },
  { id: 'C007', name: 'Thomas Reid', company: 'Micro Solutions', email: 'thomas@microsolutions.com', phone: '+1-555-1007', address: '200 Tech Park, Boston, MA', totalOrders: 8, totalSpent: 22100, status: 'inactive', lastOrderDate: '2025-12-15', segment: 'small-business' },
  { id: 'C008', name: 'Angela Foster', company: 'Pinnacle Group', email: 'angela@pinnacle.com', phone: '+1-555-1008', address: '350 Corp Center, Miami, FL', totalOrders: 55, totalSpent: 445600, status: 'active', lastOrderDate: '2026-05-12', segment: 'enterprise' },
];

export const salesOrders: SalesOrder[] = [
  { id: 'SO001', orderNumber: 'SO-2026-089', customerId: 'C001', customerName: 'Acme Corporation', items: [{ productId: 'P001', productName: 'Industrial Widget A', quantity: 200, unitPrice: 29.99, total: 5998 }, { productId: 'P003', productName: 'Gadget Pro X1', quantity: 50, unitPrice: 199.99, total: 9999.50 }], subtotal: 15997.50, tax: 1279.80, total: 17277.30, status: 'delivered', orderDate: '2026-05-01', deliveryDate: '2026-05-14', paymentStatus: 'paid' },
  { id: 'SO002', orderNumber: 'SO-2026-090', customerId: 'C003', customerName: 'Global Dynamics', items: [{ productId: 'P007', productName: 'Assembly Kit Pro', quantity: 10, unitPrice: 349.99, total: 3499.90 }, { productId: 'P009', productName: 'Precision Tool Set', quantity: 5, unitPrice: 599.99, total: 2999.95 }], subtotal: 6499.85, tax: 519.99, total: 7019.84, status: 'shipped', orderDate: '2026-05-10', deliveryDate: '2026-05-20', paymentStatus: 'paid' },
  { id: 'SO003', orderNumber: 'SO-2026-091', customerId: 'C006', customerName: 'Summit Industries', items: [{ productId: 'P005', productName: 'Component Alpha', quantity: 5000, unitPrice: 8.50, total: 42500 }], subtotal: 42500, tax: 3400, total: 45900, status: 'confirmed', orderDate: '2026-05-15', paymentStatus: 'unpaid' },
  { id: 'SO004', orderNumber: 'SO-2026-092', customerId: 'C002', customerName: 'TechFlow Inc', items: [{ productId: 'P003', productName: 'Gadget Pro X1', quantity: 25, unitPrice: 199.99, total: 4999.75 }, { productId: 'P010', productName: 'Safety Equipment Pack', quantity: 30, unitPrice: 129.99, total: 3899.70 }], subtotal: 8899.45, tax: 711.96, total: 9611.41, status: 'confirmed', orderDate: '2026-05-16', paymentStatus: 'partial' },
  { id: 'SO005', orderNumber: 'SO-2026-093', customerId: 'C008', customerName: 'Pinnacle Group', items: [{ productId: 'P008', productName: 'Assembly Kit Standard', quantity: 100, unitPrice: 189.99, total: 18999 }], subtotal: 18999, tax: 1519.92, total: 20518.92, status: 'draft', orderDate: '2026-05-18', paymentStatus: 'unpaid' },
  { id: 'SO006', orderNumber: 'SO-2026-094', customerId: 'C005', customerName: 'Pacific Trade Co', items: [{ productId: 'P011', productName: 'Steel Rods (Bundle)', quantity: 50, unitPrice: 89.99, total: 4499.50 }, { productId: 'P006', productName: 'Component Beta', quantity: 2000, unitPrice: 12.75, total: 25500 }], subtotal: 29999.50, tax: 2399.96, total: 32399.46, status: 'shipped', orderDate: '2026-05-08', deliveryDate: '2026-05-22', paymentStatus: 'paid' },
];

export const salesPipeline: SalesPipeline[] = [
  { id: 'SP001', dealName: 'Acme Q3 Expansion', customer: 'Acme Corporation', value: 125000, stage: 'negotiation', probability: 75, expectedClose: '2026-06-30', assignedTo: 'James Miller' },
  { id: 'SP002', dealName: 'TechFlow Annual Contract', customer: 'TechFlow Inc', value: 85000, stage: 'proposal', probability: 60, expectedClose: '2026-07-15', assignedTo: 'Jessica Taylor' },
  { id: 'SP003', dealName: 'Global Dynamics Enterprise', customer: 'Global Dynamics', value: 310000, stage: 'qualified', probability: 40, expectedClose: '2026-08-01', assignedTo: 'James Miller' },
  { id: 'SP004', dealName: 'Summit Manufacturing Deal', customer: 'Summit Industries', value: 175000, stage: 'closed-won', probability: 100, expectedClose: '2026-05-15', assignedTo: 'Michael Torres' },
  { id: 'SP005', dealName: 'Pinnacle New Line', customer: 'Pinnacle Group', value: 220000, stage: 'negotiation', probability: 70, expectedClose: '2026-06-15', assignedTo: 'Jessica Taylor' },
  { id: 'SP006', dealName: 'Pacific Trade Renewal', customer: 'Pacific Trade Co', value: 95000, stage: 'proposal', probability: 55, expectedClose: '2026-07-01', assignedTo: 'Michael Torres' },
  { id: 'SP007', dealName: 'StartUp Hub Pilot', customer: 'StartUp Hub', value: 15000, stage: 'lead', probability: 20, expectedClose: '2026-09-01', assignedTo: 'Michael Torres' },
  { id: 'SP008', dealName: 'Micro Solutions Upsell', customer: 'Micro Solutions', value: 42000, stage: 'closed-lost', probability: 0, expectedClose: '2026-04-30', assignedTo: 'Jessica Taylor' },
];

// ─── Procurement ───

export const suppliers: Supplier[] = [
  { id: 'S001', name: 'SteelWorks Global', contactPerson: 'Han Wei', email: 'han@steelworks.com', phone: '+1-555-2001', address: '100 Industrial Blvd, Pittsburgh, PA', rating: 4.8, totalOrders: 120, status: 'active', paymentTerms: 'Net 30', leadTimeDays: 14 },
  { id: 'S002', name: 'Component Direct', contactPerson: 'Patricia Owens', email: 'patricia@componentdirect.com', phone: '+1-555-2002', address: '250 Supply Chain Dr, Dallas, TX', rating: 4.5, totalOrders: 85, status: 'active', paymentTerms: 'Net 45', leadTimeDays: 7 },
  { id: 'S003', name: 'TechParts Co', contactPerson: 'Greg Nelson', email: 'greg@techparts.com', phone: '+1-555-2003', address: '180 Tech Way, San Jose, CA', rating: 4.2, totalOrders: 62, status: 'active', paymentTerms: 'Net 30', leadTimeDays: 10 },
  { id: 'S004', name: 'Pacific Materials', contactPerson: 'Yuki Tanaka', email: 'yuki@pacificmats.com', phone: '+1-555-2004', address: '90 Harbor Rd, Long Beach, CA', rating: 3.9, totalOrders: 38, status: 'active', paymentTerms: 'Net 60', leadTimeDays: 21 },
  { id: 'S005', name: 'Atlas Safety Supplies', contactPerson: 'Mark Fischer', email: 'mark@atlassafety.com', phone: '+1-555-2005', address: '44 Safety Lane, Columbus, OH', rating: 4.6, totalOrders: 55, status: 'active', paymentTerms: 'Net 30', leadTimeDays: 5 },
  { id: 'S006', name: 'QuickShip Electronics', contactPerson: 'Diana Flores', email: 'diana@quickship.com', phone: '+1-555-2006', address: '330 Express Ct, Memphis, TN', rating: 3.5, totalOrders: 22, status: 'inactive', paymentTerms: 'Net 15', leadTimeDays: 3 },
];

export const purchaseOrders: PurchaseOrder[] = [
  { id: 'PO001', poNumber: 'PO-2026-042', supplierId: 'S001', supplierName: 'SteelWorks Global', items: [{ productId: 'P011', productName: 'Steel Rods (Bundle)', quantity: 500, unitCost: 42.00, total: 21000 }], subtotal: 21000, tax: 1680, total: 22680, status: 'received', orderDate: '2026-04-25', expectedDelivery: '2026-05-09', receivedDate: '2026-05-10' },
  { id: 'PO002', poNumber: 'PO-2026-043', supplierId: 'S002', supplierName: 'Component Direct', items: [{ productId: 'P005', productName: 'Component Alpha', quantity: 5000, unitCost: 3.20, total: 16000 }, { productId: 'P006', productName: 'Component Beta', quantity: 3000, unitCost: 5.80, total: 17400 }], subtotal: 33400, tax: 2672, total: 36072, status: 'received', orderDate: '2026-05-01', expectedDelivery: '2026-05-08', receivedDate: '2026-05-08' },
  { id: 'PO003', poNumber: 'PO-2026-044', supplierId: 'S003', supplierName: 'TechParts Co', items: [{ productId: 'P003', productName: 'Gadget Pro X1', quantity: 200, unitCost: 85.00, total: 17000 }], subtotal: 17000, tax: 1360, total: 18360, status: 'confirmed', orderDate: '2026-05-12', expectedDelivery: '2026-05-22' },
  { id: 'PO004', poNumber: 'PO-2026-045', supplierId: 'S002', supplierName: 'Component Direct', items: [{ productId: 'P005', productName: 'Component Alpha', quantity: 3000, unitCost: 3.20, total: 9600 }], subtotal: 9600, tax: 768, total: 10368, status: 'sent', orderDate: '2026-05-15', expectedDelivery: '2026-05-22' },
  { id: 'PO005', poNumber: 'PO-2026-046', supplierId: 'S005', supplierName: 'Atlas Safety Supplies', items: [{ productId: 'P010', productName: 'Safety Equipment Pack', quantity: 200, unitCost: 55.00, total: 11000 }], subtotal: 11000, tax: 880, total: 11880, status: 'draft', orderDate: '2026-05-18', expectedDelivery: '2026-05-23' },
];

// ─── Accounting ───

export const accounts: Account[] = [
  { id: 'A001', code: '1000', name: 'Cash and Cash Equivalents', type: 'asset', balance: 1245000, isActive: true },
  { id: 'A002', code: '1100', name: 'Accounts Receivable', type: 'asset', balance: 385200, isActive: true },
  { id: 'A003', code: '1200', name: 'Inventory', type: 'asset', balance: 892400, isActive: true },
  { id: 'A004', code: '1300', name: 'Prepaid Expenses', type: 'asset', balance: 45000, isActive: true },
  { id: 'A005', code: '1500', name: 'Fixed Assets', type: 'asset', balance: 2100000, isActive: true },
  { id: 'A006', code: '1510', name: 'Accumulated Depreciation', type: 'asset', balance: -420000, isActive: true },
  { id: 'A007', code: '2000', name: 'Accounts Payable', type: 'liability', balance: 198500, isActive: true },
  { id: 'A008', code: '2100', name: 'Accrued Liabilities', type: 'liability', balance: 125000, isActive: true },
  { id: 'A009', code: '2200', name: 'Short-term Debt', type: 'liability', balance: 300000, isActive: true },
  { id: 'A010', code: '2500', name: 'Long-term Debt', type: 'liability', balance: 750000, isActive: true },
  { id: 'A011', code: '3000', name: 'Common Stock', type: 'equity', balance: 1500000, isActive: true },
  { id: 'A012', code: '3100', name: 'Retained Earnings', type: 'equity', balance: 874100, isActive: true },
  { id: 'A013', code: '4000', name: 'Product Revenue', type: 'revenue', balance: 4250000, isActive: true },
  { id: 'A014', code: '4100', name: 'Service Revenue', type: 'revenue', balance: 850000, isActive: true },
  { id: 'A015', code: '5000', name: 'Cost of Goods Sold', type: 'expense', balance: 2125000, isActive: true },
  { id: 'A016', code: '6000', name: 'Salaries & Wages', type: 'expense', balance: 1450000, isActive: true },
  { id: 'A017', code: '6100', name: 'Rent & Utilities', type: 'expense', balance: 180000, isActive: true },
  { id: 'A018', code: '6200', name: 'Marketing & Advertising', type: 'expense', balance: 320000, isActive: true },
  { id: 'A019', code: '6300', name: 'Office Supplies & Equipment', type: 'expense', balance: 75000, isActive: true },
  { id: 'A020', code: '6400', name: 'Insurance', type: 'expense', balance: 48000, isActive: true },
];

export const journalEntries: JournalEntry[] = [
  { id: 'JE001', entryNumber: 'JE-2026-0154', date: '2026-05-01', description: 'Record monthly rent payment', lines: [{ accountId: 'A017', accountName: 'Rent & Utilities', debit: 15000, credit: 0 }, { accountId: 'A001', accountName: 'Cash and Cash Equivalents', debit: 0, credit: 15000 }], status: 'posted', createdBy: 'Robert Kim' },
  { id: 'JE002', entryNumber: 'JE-2026-0155', date: '2026-05-05', description: 'Sales revenue recognition - Acme Corporation', lines: [{ accountId: 'A002', accountName: 'Accounts Receivable', debit: 17277.30, credit: 0 }, { accountId: 'A013', accountName: 'Product Revenue', debit: 0, credit: 15997.50 }, { accountId: 'A008', accountName: 'Accrued Liabilities', debit: 0, credit: 1279.80 }], status: 'posted', createdBy: 'Amanda White' },
  { id: 'JE003', entryNumber: 'JE-2026-0156', date: '2026-05-10', description: 'Inventory receipt from SteelWorks Global', lines: [{ accountId: 'A003', accountName: 'Inventory', debit: 21000, credit: 0 }, { accountId: 'A007', accountName: 'Accounts Payable', debit: 0, credit: 21000 }], status: 'posted', createdBy: 'Amanda White' },
  { id: 'JE004', entryNumber: 'JE-2026-0157', date: '2026-05-15', description: 'Payroll processing for May 1-15', lines: [{ accountId: 'A016', accountName: 'Salaries & Wages', debit: 142500, credit: 0 }, { accountId: 'A001', accountName: 'Cash and Cash Equivalents', debit: 0, credit: 142500 }], status: 'posted', createdBy: 'Robert Kim' },
  { id: 'JE005', entryNumber: 'JE-2026-0158', date: '2026-05-18', description: 'Marketing campaign expense', lines: [{ accountId: 'A018', accountName: 'Marketing & Advertising', debit: 25000, credit: 0 }, { accountId: 'A001', accountName: 'Cash and Cash Equivalents', debit: 0, credit: 25000 }], status: 'draft', createdBy: 'Emily Davis' },
];

export const invoices: Invoice[] = [
  { id: 'INV001', invoiceNumber: 'INV-2026-0321', customerId: 'C001', customerName: 'Acme Corporation', issueDate: '2026-05-01', dueDate: '2026-05-31', items: [{ description: 'Industrial Widget A x200', quantity: 200, rate: 29.99, amount: 5998 }, { description: 'Gadget Pro X1 x50', quantity: 50, rate: 199.99, amount: 9999.50 }], subtotal: 15997.50, tax: 1279.80, total: 17277.30, amountPaid: 17277.30, status: 'paid' },
  { id: 'INV002', invoiceNumber: 'INV-2026-0322', customerId: 'C003', customerName: 'Global Dynamics', issueDate: '2026-05-10', dueDate: '2026-06-09', items: [{ description: 'Assembly Kit Pro x10', quantity: 10, rate: 349.99, amount: 3499.90 }, { description: 'Precision Tool Set x5', quantity: 5, rate: 599.99, amount: 2999.95 }], subtotal: 6499.85, tax: 519.99, total: 7019.84, amountPaid: 7019.84, status: 'paid' },
  { id: 'INV003', invoiceNumber: 'INV-2026-0323', customerId: 'C006', customerName: 'Summit Industries', issueDate: '2026-05-15', dueDate: '2026-06-14', items: [{ description: 'Component Alpha x5000', quantity: 5000, rate: 8.50, amount: 42500 }], subtotal: 42500, tax: 3400, total: 45900, amountPaid: 0, status: 'sent' },
  { id: 'INV004', invoiceNumber: 'INV-2026-0324', customerId: 'C002', customerName: 'TechFlow Inc', issueDate: '2026-05-16', dueDate: '2026-06-15', items: [{ description: 'Gadget Pro X1 x25', quantity: 25, rate: 199.99, amount: 4999.75 }, { description: 'Safety Equipment Pack x30', quantity: 30, rate: 129.99, amount: 3899.70 }], subtotal: 8899.45, tax: 711.96, total: 9611.41, amountPaid: 5000, status: 'sent' },
  { id: 'INV005', invoiceNumber: 'INV-2026-0325', customerId: 'C008', customerName: 'Pinnacle Group', issueDate: '2026-04-01', dueDate: '2026-04-30', items: [{ description: 'Assembly Kit Standard x50', quantity: 50, rate: 189.99, amount: 9499.50 }], subtotal: 9499.50, tax: 759.96, total: 10259.46, amountPaid: 0, status: 'overdue' },
];

// ─── Projects ───

export const projects: Project[] = [
  { id: 'PJ001', name: 'ERP System Implementation', description: 'Deploy new ERP across all departments', client: 'Internal', manager: 'Alex Turner', status: 'active', priority: 'high', startDate: '2026-01-15', endDate: '2026-09-30', budget: 450000, spent: 185000, progress: 42, teamMembers: ['Sarah Chen', 'John Park', 'Rachel Lee', 'Nina Patel'] },
  { id: 'PJ002', name: 'Website Redesign', description: 'Complete overhaul of company website with modern design', client: 'Internal', manager: 'Emily Davis', status: 'active', priority: 'medium', startDate: '2026-03-01', endDate: '2026-07-15', budget: 120000, spent: 78000, progress: 65, teamMembers: ['Rachel Lee', 'Sophia Nguyen', 'Daniel Johnson'] },
  { id: 'PJ003', name: 'Acme Corp Integration', description: 'Custom integration for Acme Corporation systems', client: 'Acme Corporation', manager: 'Sarah Chen', status: 'active', priority: 'high', startDate: '2026-04-01', endDate: '2026-08-30', budget: 280000, spent: 62000, progress: 25, teamMembers: ['John Park', 'Brian Harris', 'Daniel Johnson'] },
  { id: 'PJ004', name: 'Mobile App v2', description: 'Major update to mobile application', client: 'Internal', manager: 'Nina Patel', status: 'planning', priority: 'medium', startDate: '2026-06-01', endDate: '2026-12-15', budget: 350000, spent: 0, progress: 0, teamMembers: ['Rachel Lee', 'Daniel Johnson'] },
  { id: 'PJ005', name: 'Data Analytics Platform', description: 'Build internal analytics and reporting platform', client: 'Internal', manager: 'Alex Turner', status: 'on-hold', priority: 'low', startDate: '2026-02-01', endDate: '2026-06-30', budget: 180000, spent: 95000, progress: 55, teamMembers: ['Brian Harris', 'John Park'] },
  { id: 'PJ006', name: 'Summit Manufacturing Portal', description: 'Customer portal for Summit Industries', client: 'Summit Industries', manager: 'Sarah Chen', status: 'completed', priority: 'high', startDate: '2025-10-01', endDate: '2026-03-31', budget: 200000, spent: 192000, progress: 100, teamMembers: ['John Park', 'Rachel Lee', 'Brian Harris'] },
];

export const tasks: Task[] = [
  { id: 'T001', projectId: 'PJ001', title: 'Database schema design', description: 'Design core database schema for ERP modules', assignee: 'John Park', status: 'done', priority: 'high', dueDate: '2026-02-15', estimatedHours: 40, loggedHours: 38 },
  { id: 'T002', projectId: 'PJ001', title: 'HR module development', description: 'Build employee management module', assignee: 'Rachel Lee', status: 'done', priority: 'high', dueDate: '2026-04-01', estimatedHours: 120, loggedHours: 115 },
  { id: 'T003', projectId: 'PJ001', title: 'Inventory module development', description: 'Build inventory and warehouse management', assignee: 'Daniel Johnson', status: 'in-progress', priority: 'high', dueDate: '2026-06-15', estimatedHours: 160, loggedHours: 72 },
  { id: 'T004', projectId: 'PJ001', title: 'Accounting integration', description: 'Integrate accounting module with existing systems', assignee: 'John Park', status: 'todo', priority: 'medium', dueDate: '2026-07-30', estimatedHours: 80, loggedHours: 0 },
  { id: 'T005', projectId: 'PJ002', title: 'UI/UX wireframes', description: 'Create wireframes for all pages', assignee: 'Sophia Nguyen', status: 'done', priority: 'high', dueDate: '2026-03-20', estimatedHours: 60, loggedHours: 55 },
  { id: 'T006', projectId: 'PJ002', title: 'Frontend development', description: 'Implement new design in React', assignee: 'Rachel Lee', status: 'in-progress', priority: 'high', dueDate: '2026-06-01', estimatedHours: 200, loggedHours: 140 },
  { id: 'T007', projectId: 'PJ002', title: 'Content migration', description: 'Migrate existing content to new platform', assignee: 'Daniel Johnson', status: 'review', priority: 'medium', dueDate: '2026-06-15', estimatedHours: 40, loggedHours: 35 },
  { id: 'T008', projectId: 'PJ003', title: 'API specification', description: 'Define integration API endpoints', assignee: 'Brian Harris', status: 'done', priority: 'high', dueDate: '2026-04-15', estimatedHours: 30, loggedHours: 28 },
  { id: 'T009', projectId: 'PJ003', title: 'Authentication setup', description: 'OAuth2 integration for secure access', assignee: 'Brian Harris', status: 'in-progress', priority: 'high', dueDate: '2026-05-30', estimatedHours: 50, loggedHours: 22 },
  { id: 'T010', projectId: 'PJ003', title: 'Data sync engine', description: 'Build bidirectional data synchronization', assignee: 'John Park', status: 'todo', priority: 'critical', dueDate: '2026-06-30', estimatedHours: 120, loggedHours: 0 },
];

// ─── Dashboard Aggregated Data ───

export const revenueByMonth = [
  { month: 'Jan', revenue: 380000, expenses: 295000, profit: 85000 },
  { month: 'Feb', revenue: 420000, expenses: 310000, profit: 110000 },
  { month: 'Mar', revenue: 395000, expenses: 305000, profit: 90000 },
  { month: 'Apr', revenue: 510000, expenses: 340000, profit: 170000 },
  { month: 'May', revenue: 485000, expenses: 325000, profit: 160000 },
  { month: 'Jun', revenue: 0, expenses: 0, profit: 0 },
];

export const salesByCategory = [
  { category: 'Widgets', value: 285000 },
  { category: 'Gadgets', value: 420000 },
  { category: 'Components', value: 195000 },
  { category: 'Assemblies', value: 310000 },
  { category: 'Tools', value: 145000 },
  { category: 'Safety', value: 88000 },
  { category: 'Raw Materials', value: 165000 },
];

export const departmentHeadcount = [
  { department: 'Engineering', count: 42 },
  { department: 'Sales', count: 28 },
  { department: 'Operations', count: 20 },
  { department: 'Customer Support', count: 18 },
  { department: 'Marketing', count: 15 },
  { department: 'Finance', count: 12 },
  { department: 'Product', count: 10 },
  { department: 'HR', count: 8 },
];

export const recentActivity = [
  { id: 1, action: 'Sales order SO-2026-093 created', module: 'Sales', user: 'Jessica Taylor', time: '2 hours ago' },
  { id: 2, action: 'Purchase order PO-2026-046 drafted', module: 'Procurement', user: 'Chris Anderson', time: '3 hours ago' },
  { id: 3, action: 'Employee leave request approved', module: 'HR', user: 'Maria Garcia', time: '4 hours ago' },
  { id: 4, action: 'Invoice INV-2026-0324 sent to TechFlow Inc', module: 'Accounting', user: 'Amanda White', time: '5 hours ago' },
  { id: 5, action: 'Stock transfer TF-2026-012 completed', module: 'Inventory', user: 'Chris Anderson', time: '6 hours ago' },
  { id: 6, action: 'Journal entry JE-2026-0158 created', module: 'Accounting', user: 'Emily Davis', time: '8 hours ago' },
  { id: 7, action: 'New deal added to pipeline: Pinnacle New Line', module: 'Sales', user: 'Jessica Taylor', time: '1 day ago' },
  { id: 8, action: 'Project milestone completed: HR Module', module: 'Projects', user: 'Rachel Lee', time: '1 day ago' },
];
