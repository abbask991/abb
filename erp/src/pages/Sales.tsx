import { useState } from 'react';
import {
  Plus, Download, Search, DollarSign, Users as UsersIcon,
  ShoppingCart, TrendingUp,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import PageHeader from '../components/PageHeader';
import Tabs from '../components/Tabs';
import StatusBadge from '../components/StatusBadge';
import KpiCard from '../components/KpiCard';
import { formatCurrency, formatDate, tooltipCurrencyFormatter } from '../components/FormatHelpers';
import { customers, salesOrders, salesPipeline } from '../data/mockData';

const STAGE_COLORS: Record<string, string> = {
  lead: '#94a3b8',
  qualified: '#0891b2',
  proposal: '#d97706',
  negotiation: '#7c3aed',
  'closed-won': '#059669',
  'closed-lost': '#dc2626',
};

const pipelineStages = ['lead', 'qualified', 'proposal', 'negotiation', 'closed-won', 'closed-lost'];
const pipelineChartData = pipelineStages.map((stage) => ({
  stage: stage.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
  value: salesPipeline.filter((d) => d.stage === stage).reduce((s, d) => s + d.value, 0),
  count: salesPipeline.filter((d) => d.stage === stage).length,
}));

const segmentData = [
  { name: 'Enterprise', value: customers.filter((c) => c.segment === 'enterprise').length },
  { name: 'Mid-Market', value: customers.filter((c) => c.segment === 'mid-market').length },
  { name: 'Small Business', value: customers.filter((c) => c.segment === 'small-business').length },
];
const SEG_COLORS = ['#2563eb', '#d97706', '#059669'];

const tabs = [
  { id: 'orders', label: 'Sales Orders', count: salesOrders.length },
  { id: 'customers', label: 'Customers', count: customers.length },
  { id: 'pipeline', label: 'Pipeline', count: salesPipeline.length },
];

export default function Sales() {
  const [activeTab, setActiveTab] = useState('orders');
  const [search, setSearch] = useState('');

  const totalSales = salesOrders.reduce((s, o) => s + o.total, 0);
  const avgOrderValue = totalSales / salesOrders.length;
  const pipelineValue = salesPipeline.filter((d) => !['closed-won', 'closed-lost'].includes(d.stage)).reduce((s, d) => s + d.value, 0);

  return (
    <div>
      <PageHeader
        title="Sales"
        subtitle="Manage sales orders, customers, and your pipeline."
        actions={
          <>
            <button className="btn-secondary"><Download className="w-4 h-4" /> Export</button>
            <button className="btn-primary"><Plus className="w-4 h-4" /> New Order</button>
          </>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard title="Total Sales" value={formatCurrency(totalSales)} change="+15.3%" trend="up" icon={DollarSign} color="blue" />
        <KpiCard title="Total Customers" value={String(customers.length)} change="+2" trend="up" icon={UsersIcon} color="emerald" />
        <KpiCard title="Avg. Order Value" value={formatCurrency(avgOrderValue)} icon={ShoppingCart} color="amber" />
        <KpiCard title="Pipeline Value" value={formatCurrency(pipelineValue)} change="+8.2%" trend="up" icon={TrendingUp} color="purple" />
      </div>

      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === 'orders' && (
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">Sales Orders</h3>
            <div className="relative">
              <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
              <input type="text" placeholder="Search orders..." value={search} onChange={(e) => setSearch(e.target.value)} className="input pl-9 w-64" />
            </div>
          </div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Order #</th>
                  <th>Customer</th>
                  <th>Items</th>
                  <th>Total</th>
                  <th>Order Date</th>
                  <th>Delivery</th>
                  <th>Payment</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {salesOrders
                  .filter((o) =>
                    o.orderNumber.toLowerCase().includes(search.toLowerCase()) ||
                    o.customerName.toLowerCase().includes(search.toLowerCase())
                  )
                  .map((o) => (
                    <tr key={o.id}>
                      <td className="font-mono text-sm font-medium">{o.orderNumber}</td>
                      <td className="font-medium">{o.customerName}</td>
                      <td className="text-gray-500">{o.items.length} item(s)</td>
                      <td className="font-bold">{formatCurrency(o.total)}</td>
                      <td>{formatDate(o.orderDate)}</td>
                      <td>{o.deliveryDate ? formatDate(o.deliveryDate) : '-'}</td>
                      <td><StatusBadge status={o.paymentStatus} /></td>
                      <td><StatusBadge status={o.status} /></td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'customers' && (
        <div>
          <div className="card mb-6">
            <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Customer Segments</h3></div>
            <div className="card-body flex items-center justify-center">
              <ResponsiveContainer width="100%" height={220}>
                <PieChart>
                  <Pie data={segmentData} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={3} dataKey="value" nameKey="name">
                    {segmentData.map((_, i) => <Cell key={i} fill={SEG_COLORS[i]} />)}
                  </Pie>
                  <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card">
            <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Customer Directory</h3></div>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>Customer</th>
                    <th>Company</th>
                    <th>Segment</th>
                    <th>Total Orders</th>
                    <th>Total Spent</th>
                    <th>Last Order</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {customers.map((c) => (
                    <tr key={c.id}>
                      <td>
                        <div>
                          <p className="font-medium">{c.name}</p>
                          <p className="text-xs text-gray-500">{c.email}</p>
                        </div>
                      </td>
                      <td>{c.company}</td>
                      <td><StatusBadge status={c.segment} /></td>
                      <td className="text-center">{c.totalOrders}</td>
                      <td className="font-medium">{formatCurrency(c.totalSpent)}</td>
                      <td>{formatDate(c.lastOrderDate)}</td>
                      <td><StatusBadge status={c.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'pipeline' && (
        <div>
          <div className="card mb-6">
            <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Pipeline by Stage</h3></div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={pipelineChartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="stage" tick={{ fontSize: 11 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v / 1000}k`} />
                  <Tooltip formatter={tooltipCurrencyFormatter} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={40} name="Value">
                    {pipelineChartData.map((_, i) => (
                      <Cell key={i} fill={STAGE_COLORS[pipelineStages[i]] || '#94a3b8'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card">
            <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">All Deals</h3></div>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>Deal</th>
                    <th>Customer</th>
                    <th>Value</th>
                    <th>Stage</th>
                    <th>Probability</th>
                    <th>Expected Close</th>
                    <th>Assigned To</th>
                  </tr>
                </thead>
                <tbody>
                  {salesPipeline.map((d) => (
                    <tr key={d.id}>
                      <td className="font-medium">{d.dealName}</td>
                      <td>{d.customer}</td>
                      <td className="font-bold">{formatCurrency(d.value)}</td>
                      <td><StatusBadge status={d.stage} /></td>
                      <td>
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-2 bg-gray-200 rounded-full">
                            <div className="h-2 bg-blue-500 rounded-full" style={{ width: `${d.probability}%` }} />
                          </div>
                          <span className="text-xs text-gray-500">{d.probability}%</span>
                        </div>
                      </td>
                      <td>{formatDate(d.expectedClose)}</td>
                      <td>{d.assignedTo}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
