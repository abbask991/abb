import { useState } from 'react';
import { Plus, Download, Search, AlertTriangle, Package, Warehouse as WarehouseIcon } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import PageHeader from '../components/PageHeader';
import Tabs from '../components/Tabs';
import StatusBadge from '../components/StatusBadge';
import KpiCard from '../components/KpiCard';
import { formatCurrency, formatNumber, formatDate } from '../components/FormatHelpers';
import { products, warehouses, stockMovements } from '../data/mockData';

const categoryData = Object.entries(
  products.reduce<Record<string, number>>((acc, p) => {
    acc[p.category] = (acc[p.category] || 0) + p.stock;
    return acc;
  }, {})
).map(([category, stock]) => ({ category, stock }));

const statusData = [
  { name: 'In Stock', value: products.filter((p) => p.status === 'in-stock').length },
  { name: 'Low Stock', value: products.filter((p) => p.status === 'low-stock').length },
  { name: 'Out of Stock', value: products.filter((p) => p.status === 'out-of-stock').length },
];
const STATUS_COLORS = ['#059669', '#d97706', '#dc2626'];

const tabs = [
  { id: 'products', label: 'Products', count: products.length },
  { id: 'warehouses', label: 'Warehouses', count: warehouses.length },
  { id: 'movements', label: 'Stock Movements', count: stockMovements.length },
];

export default function Inventory() {
  const [activeTab, setActiveTab] = useState('products');
  const [search, setSearch] = useState('');

  const totalValue = products.reduce((s, p) => s + p.price * p.stock, 0);
  const totalItems = products.reduce((s, p) => s + p.stock, 0);
  const lowStock = products.filter((p) => p.status === 'low-stock').length;
  const outOfStock = products.filter((p) => p.status === 'out-of-stock').length;

  const filteredProducts = products.filter(
    (p) =>
      p.name.toLowerCase().includes(search.toLowerCase()) ||
      p.sku.toLowerCase().includes(search.toLowerCase()) ||
      p.category.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div>
      <PageHeader
        title="Inventory Management"
        subtitle="Track products, warehouses, and stock movements."
        actions={
          <>
            <button className="btn-secondary"><Download className="w-4 h-4" /> Export</button>
            <button className="btn-primary"><Plus className="w-4 h-4" /> Add Product</button>
          </>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard title="Total Inventory Value" value={formatCurrency(totalValue)} icon={Package} color="blue" />
        <KpiCard title="Total Items in Stock" value={formatNumber(totalItems)} icon={WarehouseIcon} color="emerald" />
        <KpiCard title="Low Stock Items" value={String(lowStock)} icon={AlertTriangle} color="amber" />
        <KpiCard title="Out of Stock" value={String(outOfStock)} icon={AlertTriangle} color="red" />
      </div>

      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === 'products' && (
        <div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <div className="card lg:col-span-2">
              <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Stock by Category</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={categoryData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="category" tick={{ fontSize: 11 }} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="stock" fill="#2563eb" radius={[4, 4, 0, 0]} barSize={36} name="Units" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Stock Status</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={statusData} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={3} dataKey="value" nameKey="name">
                      {statusData.map((_, i) => <Cell key={i} fill={STATUS_COLORS[i]} />)}
                    </Pie>
                    <Legend iconType="circle" wrapperStyle={{ fontSize: 12 }} />
                    <Tooltip />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-700">Product Catalog</h3>
              <div className="relative">
                <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input type="text" placeholder="Search products..." value={search} onChange={(e) => setSearch(e.target.value)} className="input pl-9 w-64" />
              </div>
            </div>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>Product</th>
                    <th>SKU</th>
                    <th>Category</th>
                    <th>Price</th>
                    <th>Cost</th>
                    <th>Stock</th>
                    <th>Reorder Lvl</th>
                    <th>Warehouse</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredProducts.map((p) => (
                    <tr key={p.id}>
                      <td className="font-medium">{p.name}</td>
                      <td className="text-gray-500 font-mono text-xs">{p.sku}</td>
                      <td>{p.category}</td>
                      <td>{formatCurrency(p.price)}</td>
                      <td className="text-gray-500">{formatCurrency(p.cost)}</td>
                      <td className={p.stock <= p.reorderLevel ? 'text-red-600 font-bold' : ''}>{formatNumber(p.stock)}</td>
                      <td className="text-gray-500">{formatNumber(p.reorderLevel)}</td>
                      <td className="text-xs">{p.warehouse}</td>
                      <td><StatusBadge status={p.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'warehouses' && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {warehouses.map((w) => (
            <div key={w.id} className="card p-6">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-12 h-12 bg-blue-50 rounded-xl flex items-center justify-center">
                  <WarehouseIcon className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{w.name}</h3>
                  <p className="text-sm text-gray-500">{w.location}</p>
                </div>
              </div>
              <div className="space-y-3">
                <div>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-500">Utilization</span>
                    <span className="font-medium">{w.utilization}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div
                      className={`h-2 rounded-full ${w.utilization > 80 ? 'bg-red-500' : w.utilization > 60 ? 'bg-amber-500' : 'bg-emerald-500'}`}
                      style={{ width: `${w.utilization}%` }}
                    />
                  </div>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Capacity</span>
                  <span className="font-medium">{formatNumber(w.capacity)} units</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Products</span>
                  <span className="font-medium">{w.productCount}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Manager</span>
                  <span className="font-medium">{w.manager}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'movements' && (
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Stock Movements</h3></div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Product</th>
                  <th>Type</th>
                  <th>Quantity</th>
                  <th>From</th>
                  <th>To</th>
                  <th>Reference</th>
                  <th>Note</th>
                </tr>
              </thead>
              <tbody>
                {stockMovements.map((sm) => (
                  <tr key={sm.id}>
                    <td>{formatDate(sm.date)}</td>
                    <td className="font-medium">{sm.productName}</td>
                    <td>
                      <span className={`badge ${sm.type === 'in' ? 'badge-success' : sm.type === 'out' ? 'badge-danger' : sm.type === 'transfer' ? 'badge-info' : 'badge-warning'}`}>
                        {sm.type.toUpperCase()}
                      </span>
                    </td>
                    <td className={sm.quantity > 0 ? 'text-emerald-600 font-medium' : 'text-red-600 font-medium'}>
                      {sm.quantity > 0 ? `+${sm.quantity}` : sm.quantity}
                    </td>
                    <td className="text-gray-500">{sm.fromWarehouse || '-'}</td>
                    <td className="text-gray-500">{sm.toWarehouse || '-'}</td>
                    <td className="font-mono text-xs">{sm.reference}</td>
                    <td className="text-gray-500 text-xs">{sm.note}</td>
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
