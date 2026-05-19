import {
  DollarSign, Users, Package, ShoppingCart,
  TrendingUp, AlertTriangle, Clock, FileText,
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar, PieChart, Pie, Cell, Legend,
} from 'recharts';
import KpiCard from '../components/KpiCard';
import PageHeader from '../components/PageHeader';
import StatusBadge from '../components/StatusBadge';
import { formatCurrency, tooltipCurrencyFormatter } from '../components/FormatHelpers';
import {
  revenueByMonth, salesByCategory, departmentHeadcount,
  recentActivity, salesOrders, invoices, products,
} from '../data/mockData';

const PIE_COLORS = ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#db2777'];

export default function Dashboard() {
  const totalRevenue = revenueByMonth.reduce((s, m) => s + m.revenue, 0);
  const totalExpenses = revenueByMonth.reduce((s, m) => s + m.expenses, 0);
  const openOrders = salesOrders.filter((o) => !['delivered', 'cancelled'].includes(o.status)).length;
  const overdueInvoices = invoices.filter((i) => i.status === 'overdue').length;
  const lowStockItems = products.filter((p) => p.status === 'low-stock' || p.status === 'out-of-stock').length;

  return (
    <div>
      <PageHeader
        title="Dashboard"
        subtitle="Welcome back. Here's what's happening across your organization."
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
        <KpiCard
          title="Total Revenue (YTD)"
          value={formatCurrency(totalRevenue)}
          change="+12.5%"
          trend="up"
          icon={DollarSign}
          color="blue"
        />
        <KpiCard
          title="Total Employees"
          value="153"
          change="+8"
          trend="up"
          icon={Users}
          color="emerald"
        />
        <KpiCard
          title="Open Orders"
          value={String(openOrders)}
          change="-3"
          trend="down"
          icon={ShoppingCart}
          color="amber"
        />
        <KpiCard
          title="Low Stock Alerts"
          value={String(lowStockItems)}
          trend="up"
          change="+2"
          icon={Package}
          color="red"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="lg:col-span-2 card">
          <div className="card-header flex items-center justify-between">
            <h2 className="text-base font-semibold text-gray-900">Revenue vs Expenses</h2>
            <span className="text-sm text-gray-500">2026</span>
          </div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={revenueByMonth.filter((m) => m.revenue > 0)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v / 1000}k`} />
                <Tooltip formatter={tooltipCurrencyFormatter} />
                <Area type="monotone" dataKey="revenue" stackId="1" stroke="#2563eb" fill="#dbeafe" name="Revenue" />
                <Area type="monotone" dataKey="expenses" stackId="2" stroke="#dc2626" fill="#fee2e2" name="Expenses" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="text-base font-semibold text-gray-900">Sales by Category</h2>
          </div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={salesByCategory}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={2}
                  dataKey="value"
                  nameKey="category"
                >
                  {salesByCategory.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                <Tooltip formatter={tooltipCurrencyFormatter} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="text-base font-semibold text-gray-900">Department Headcount</h2>
          </div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={departmentHeadcount} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="department" tick={{ fontSize: 11 }} width={100} />
                <Tooltip />
                <Bar dataKey="count" fill="#2563eb" radius={[0, 4, 4, 0]} barSize={18} name="Employees" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="text-base font-semibold text-gray-900">Quick Stats</h2>
          </div>
          <div className="card-body space-y-4">
            <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
              <div className="flex items-center gap-3">
                <TrendingUp className="w-5 h-5 text-blue-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Net Profit (YTD)</p>
                  <p className="text-xs text-gray-500">Revenue minus expenses</p>
                </div>
              </div>
              <span className="font-bold text-blue-600">{formatCurrency(totalRevenue - totalExpenses)}</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-amber-50 rounded-lg">
              <div className="flex items-center gap-3">
                <AlertTriangle className="w-5 h-5 text-amber-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Overdue Invoices</p>
                  <p className="text-xs text-gray-500">Requires attention</p>
                </div>
              </div>
              <span className="font-bold text-amber-600">{overdueInvoices}</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-emerald-50 rounded-lg">
              <div className="flex items-center gap-3">
                <Clock className="w-5 h-5 text-emerald-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Pending POs</p>
                  <p className="text-xs text-gray-500">Awaiting delivery</p>
                </div>
              </div>
              <span className="font-bold text-emerald-600">3</span>
            </div>
            <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
              <div className="flex items-center gap-3">
                <FileText className="w-5 h-5 text-purple-600" />
                <div>
                  <p className="text-sm font-medium text-gray-900">Active Projects</p>
                  <p className="text-xs text-gray-500">Currently in progress</p>
                </div>
              </div>
              <span className="font-bold text-purple-600">3</span>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <h2 className="text-base font-semibold text-gray-900">Recent Activity</h2>
          </div>
          <div className="card-body">
            <div className="space-y-3 max-h-[340px] overflow-y-auto scrollbar-thin">
              {recentActivity.map((item) => (
                <div key={item.id} className="flex gap-3 items-start">
                  <div className="w-2 h-2 mt-2 rounded-full bg-blue-400 shrink-0" />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-gray-800 leading-snug">{item.action}</p>
                    <div className="flex items-center gap-2 mt-1">
                      <StatusBadge status={item.module} />
                      <span className="text-xs text-gray-400">{item.time}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
