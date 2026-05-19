import { Download, Printer, Calendar } from 'lucide-react';
import {
  AreaChart, Area, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from 'recharts';
import PageHeader from '../components/PageHeader';
import { formatCurrency, tooltipCurrencyFormatter } from '../components/FormatHelpers';
import {
  revenueByMonth, salesByCategory, departmentHeadcount,
  accounts,
} from '../data/mockData';

const profitMarginData = revenueByMonth.filter((m) => m.revenue > 0).map((m) => ({
  month: m.month,
  margin: Math.round((m.profit / m.revenue) * 100),
  revenue: m.revenue,
  profit: m.profit,
}));

const cashFlowData = [
  { month: 'Jan', inflow: 410000, outflow: 345000 },
  { month: 'Feb', inflow: 445000, outflow: 360000 },
  { month: 'Mar', inflow: 425000, outflow: 355000 },
  { month: 'Apr', inflow: 540000, outflow: 388000 },
  { month: 'May', inflow: 510000, outflow: 375000 },
];

const inventoryTurnover = [
  { category: 'Widgets', turnover: 4.2 },
  { category: 'Gadgets', turnover: 3.8 },
  { category: 'Components', turnover: 6.1 },
  { category: 'Assemblies', turnover: 2.9 },
  { category: 'Tools', turnover: 1.5 },
  { category: 'Safety', turnover: 3.2 },
  { category: 'Raw Materials', turnover: 5.5 },
];

export default function Reports() {
  const totalAssets = accounts.filter((a) => a.type === 'asset').reduce((s, a) => s + a.balance, 0);
  const totalLiabilities = accounts.filter((a) => a.type === 'liability').reduce((s, a) => s + a.balance, 0);
  const totalRevenue = revenueByMonth.reduce((s, m) => s + m.revenue, 0);
  const totalCOGS = accounts.find((a) => a.code === '5000')?.balance ?? 0;
  const grossProfit = totalRevenue - totalCOGS;
  const opex = accounts.filter((a) => a.type === 'expense' && a.code !== '5000').reduce((s, a) => s + a.balance, 0);

  return (
    <div>
      <PageHeader
        title="Reports & Analytics"
        subtitle="Comprehensive business intelligence across all modules."
        actions={
          <>
            <button className="btn-secondary"><Printer className="w-4 h-4" /> Print</button>
            <button className="btn-primary"><Download className="w-4 h-4" /> Export PDF</button>
          </>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Revenue & Profit Trend</h3>
            <div className="flex items-center gap-1 text-xs text-gray-500"><Calendar className="w-3 h-3" /> 2026 YTD</div>
          </div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={280}>
              <AreaChart data={revenueByMonth.filter((m) => m.revenue > 0)}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v / 1000}k`} />
                <Tooltip formatter={tooltipCurrencyFormatter} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                <Area type="monotone" dataKey="revenue" stroke="#2563eb" fill="#dbeafe" name="Revenue" />
                <Area type="monotone" dataKey="profit" stroke="#059669" fill="#d1fae5" name="Profit" />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Cash Flow Analysis</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={cashFlowData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v / 1000}k`} />
                <Tooltip formatter={tooltipCurrencyFormatter} />
                <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                <Bar dataKey="inflow" fill="#059669" radius={[4, 4, 0, 0]} barSize={24} name="Cash Inflow" />
                <Bar dataKey="outflow" fill="#dc2626" radius={[4, 4, 0, 0]} barSize={24} name="Cash Outflow" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Profit Margin Trend</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={profitMarginData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="month" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `${v}%`} />
                <Tooltip formatter={(value: unknown) => `${value}%`} />
                <Line type="monotone" dataKey="margin" stroke="#7c3aed" strokeWidth={3} dot={{ r: 5 }} name="Profit Margin %" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Inventory Turnover by Category</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={inventoryTurnover} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="category" tick={{ fontSize: 11 }} width={90} />
                <Tooltip />
                <Bar dataKey="turnover" fill="#0891b2" radius={[0, 4, 4, 0]} barSize={18} name="Turnover Ratio" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Sales by Category</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={salesByCategory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis dataKey="category" tick={{ fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v / 1000}k`} />
                <Tooltip formatter={tooltipCurrencyFormatter} />
                <Bar dataKey="value" fill="#d97706" radius={[4, 4, 0, 0]} barSize={24} name="Revenue" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Headcount by Department</h3></div>
          <div className="card-body">
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={departmentHeadcount} layout="vertical" margin={{ left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                <XAxis type="number" tick={{ fontSize: 12 }} />
                <YAxis type="category" dataKey="department" tick={{ fontSize: 10 }} width={90} />
                <Tooltip />
                <Bar dataKey="count" fill="#2563eb" radius={[0, 4, 4, 0]} barSize={16} name="Employees" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Income Statement (YTD)</h3></div>
          <div className="card-body space-y-2">
            <div className="flex justify-between text-sm py-1"><span className="text-gray-600">Total Revenue</span><span className="font-medium">{formatCurrency(totalRevenue)}</span></div>
            <div className="flex justify-between text-sm py-1"><span className="text-gray-600">Cost of Goods Sold</span><span className="font-medium text-red-600">({formatCurrency(totalCOGS)})</span></div>
            <div className="flex justify-between text-sm py-1 border-t font-bold"><span>Gross Profit</span><span>{formatCurrency(grossProfit)}</span></div>
            <div className="flex justify-between text-sm py-1"><span className="text-gray-600">Operating Expenses</span><span className="font-medium text-red-600">({formatCurrency(opex)})</span></div>
            <div className="flex justify-between text-sm py-2 border-t-2 border-gray-300 font-bold text-lg">
              <span>Net Income</span>
              <span className="text-emerald-600">{formatCurrency(grossProfit - opex)}</span>
            </div>
            <div className="mt-4 space-y-2">
              <div className="flex justify-between text-sm"><span className="text-gray-600">Gross Margin</span><span className="font-medium">{((grossProfit / totalRevenue) * 100).toFixed(1)}%</span></div>
              <div className="flex justify-between text-sm"><span className="text-gray-600">Net Margin</span><span className="font-medium">{(((grossProfit - opex) / totalRevenue) * 100).toFixed(1)}%</span></div>
              <div className="flex justify-between text-sm"><span className="text-gray-600">Current Ratio</span><span className="font-medium">{(totalAssets / totalLiabilities).toFixed(2)}</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
