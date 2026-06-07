import { useState } from 'react';
import { Plus, Download, Search, Truck, Star, DollarSign, Clock } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import PageHeader from '../components/PageHeader';
import Tabs from '../components/Tabs';
import StatusBadge from '../components/StatusBadge';
import KpiCard from '../components/KpiCard';
import { formatCurrency, formatDate, tooltipCurrencyFormatter } from '../components/FormatHelpers';
import { suppliers, purchaseOrders } from '../data/mockData';

const poByStatus = ['draft', 'sent', 'confirmed', 'received', 'cancelled'].map((s) => ({
  status: s.charAt(0).toUpperCase() + s.slice(1),
  count: purchaseOrders.filter((po) => po.status === s).length,
  value: purchaseOrders.filter((po) => po.status === s).reduce((sum, po) => sum + po.total, 0),
}));

const tabs = [
  { id: 'orders', label: 'Purchase Orders', count: purchaseOrders.length },
  { id: 'suppliers', label: 'Suppliers', count: suppliers.length },
];

export default function Procurement() {
  const [activeTab, setActiveTab] = useState('orders');
  const [search, setSearch] = useState('');

  const totalSpend = purchaseOrders.reduce((s, po) => s + po.total, 0);
  const pendingPOs = purchaseOrders.filter((po) => ['draft', 'sent', 'confirmed'].includes(po.status)).length;
  const avgLeadTime = Math.round(suppliers.reduce((s, sup) => s + sup.leadTimeDays, 0) / suppliers.length);

  return (
    <div>
      <PageHeader
        title="Procurement"
        subtitle="Manage suppliers and purchase orders."
        actions={
          <>
            <button className="btn-secondary"><Download className="w-4 h-4" /> Export</button>
            <button className="btn-primary"><Plus className="w-4 h-4" /> New PO</button>
          </>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard title="Total PO Spend" value={formatCurrency(totalSpend)} icon={DollarSign} color="blue" />
        <KpiCard title="Active Suppliers" value={String(suppliers.filter((s) => s.status === 'active').length)} icon={Truck} color="emerald" />
        <KpiCard title="Pending POs" value={String(pendingPOs)} icon={Clock} color="amber" />
        <KpiCard title="Avg Lead Time" value={`${avgLeadTime} days`} icon={Clock} color="cyan" />
      </div>

      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === 'orders' && (
        <div>
          <div className="card mb-6">
            <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">POs by Status</h3></div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={poByStatus}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="status" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} tickFormatter={(v) => `$${v / 1000}k`} />
                  <Tooltip formatter={tooltipCurrencyFormatter} />
                  <Bar dataKey="value" fill="#2563eb" radius={[4, 4, 0, 0]} barSize={40} name="Value" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="card">
            <div className="card-header flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-700">Purchase Orders</h3>
              <div className="relative">
                <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input type="text" placeholder="Search POs..." value={search} onChange={(e) => setSearch(e.target.value)} className="input pl-9 w-64" />
              </div>
            </div>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>PO Number</th>
                    <th>Supplier</th>
                    <th>Items</th>
                    <th>Total</th>
                    <th>Order Date</th>
                    <th>Expected Delivery</th>
                    <th>Received</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {purchaseOrders
                    .filter((po) =>
                      po.poNumber.toLowerCase().includes(search.toLowerCase()) ||
                      po.supplierName.toLowerCase().includes(search.toLowerCase())
                    )
                    .map((po) => (
                      <tr key={po.id}>
                        <td className="font-mono text-sm font-medium">{po.poNumber}</td>
                        <td className="font-medium">{po.supplierName}</td>
                        <td className="text-gray-500">{po.items.length} item(s)</td>
                        <td className="font-bold">{formatCurrency(po.total)}</td>
                        <td>{formatDate(po.orderDate)}</td>
                        <td>{formatDate(po.expectedDelivery)}</td>
                        <td>{po.receivedDate ? formatDate(po.receivedDate) : '-'}</td>
                        <td><StatusBadge status={po.status} /></td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'suppliers' && (
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Supplier Directory</h3></div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Supplier</th>
                  <th>Contact</th>
                  <th>Rating</th>
                  <th>Total Orders</th>
                  <th>Payment Terms</th>
                  <th>Lead Time</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {suppliers.map((s) => (
                  <tr key={s.id}>
                    <td>
                      <div>
                        <p className="font-medium">{s.name}</p>
                        <p className="text-xs text-gray-500">{s.address}</p>
                      </div>
                    </td>
                    <td>
                      <div>
                        <p className="text-sm">{s.contactPerson}</p>
                        <p className="text-xs text-gray-500">{s.email}</p>
                      </div>
                    </td>
                    <td>
                      <div className="flex items-center gap-1">
                        <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
                        <span className="font-medium">{s.rating}</span>
                      </div>
                    </td>
                    <td className="text-center">{s.totalOrders}</td>
                    <td>{s.paymentTerms}</td>
                    <td>{s.leadTimeDays} days</td>
                    <td><StatusBadge status={s.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
