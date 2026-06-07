import { useState } from 'react';
import { Building2, User, Shield, Bell, Globe, Database, Save } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import Tabs from '../components/Tabs';

const tabs = [
  { id: 'company', label: 'Company' },
  { id: 'users', label: 'Users & Roles' },
  { id: 'notifications', label: 'Notifications' },
  { id: 'system', label: 'System' },
];

const users = [
  { name: 'Admin User', email: 'admin@company.com', role: 'Super Admin', lastActive: '2 minutes ago', status: 'online' },
  { name: 'Sarah Chen', email: 'sarah.chen@company.com', role: 'Manager', lastActive: '15 minutes ago', status: 'online' },
  { name: 'Robert Kim', email: 'robert.kim@company.com', role: 'Finance Admin', lastActive: '1 hour ago', status: 'away' },
  { name: 'James Miller', email: 'james.miller@company.com', role: 'Sales Manager', lastActive: '3 hours ago', status: 'offline' },
  { name: 'Maria Garcia', email: 'maria.garcia@company.com', role: 'HR Manager', lastActive: '30 minutes ago', status: 'online' },
];

export default function Settings() {
  const [activeTab, setActiveTab] = useState('company');

  return (
    <div>
      <PageHeader
        title="Settings"
        subtitle="Manage your organization and system configuration."
      />

      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === 'company' && (
        <div className="max-w-3xl">
          <div className="card mb-6">
            <div className="card-header flex items-center gap-2">
              <Building2 className="w-4 h-4 text-gray-500" />
              <h3 className="text-sm font-semibold text-gray-700">Company Information</h3>
            </div>
            <div className="card-body space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Company Name</label>
                  <input type="text" defaultValue="Acme Industries Inc." className="input" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Industry</label>
                  <select className="select">
                    <option>Manufacturing</option>
                    <option>Technology</option>
                    <option>Retail</option>
                    <option>Services</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Tax ID</label>
                  <input type="text" defaultValue="XX-XXXXXXX" className="input" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Currency</label>
                  <select className="select">
                    <option>USD - US Dollar</option>
                    <option>EUR - Euro</option>
                    <option>GBP - British Pound</option>
                  </select>
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Address</label>
                <textarea defaultValue="100 Enterprise Blvd, Houston, TX 77001" className="input" rows={2} />
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Phone</label>
                  <input type="tel" defaultValue="+1-555-0100" className="input" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Website</label>
                  <input type="url" defaultValue="https://www.acme-industries.com" className="input" />
                </div>
              </div>
              <div className="flex justify-end pt-2">
                <button className="btn-primary"><Save className="w-4 h-4" /> Save Changes</button>
              </div>
            </div>
          </div>

          <div className="card">
            <div className="card-header flex items-center gap-2">
              <Globe className="w-4 h-4 text-gray-500" />
              <h3 className="text-sm font-semibold text-gray-700">Localization</h3>
            </div>
            <div className="card-body space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Language</label>
                  <select className="select">
                    <option>English (US)</option>
                    <option>Spanish</option>
                    <option>French</option>
                    <option>German</option>
                    <option>Arabic</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Timezone</label>
                  <select className="select">
                    <option>America/Chicago (CST)</option>
                    <option>America/New_York (EST)</option>
                    <option>America/Los_Angeles (PST)</option>
                    <option>Europe/London (GMT)</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Date Format</label>
                  <select className="select">
                    <option>MM/DD/YYYY</option>
                    <option>DD/MM/YYYY</option>
                    <option>YYYY-MM-DD</option>
                  </select>
                </div>
              </div>
              <div className="flex justify-end pt-2">
                <button className="btn-primary"><Save className="w-4 h-4" /> Save Changes</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'users' && (
        <div>
          <div className="card mb-6">
            <div className="card-header flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Shield className="w-4 h-4 text-gray-500" />
                <h3 className="text-sm font-semibold text-gray-700">Users & Access Control</h3>
              </div>
              <button className="btn-primary text-sm"><User className="w-4 h-4" /> Add User</button>
            </div>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Role</th>
                    <th>Last Active</th>
                    <th>Status</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {users.map((user, i) => (
                    <tr key={i}>
                      <td>
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-sm font-medium text-blue-700">
                            {user.name.split(' ').map((n) => n[0]).join('')}
                          </div>
                          <div>
                            <p className="font-medium">{user.name}</p>
                            <p className="text-xs text-gray-500">{user.email}</p>
                          </div>
                        </div>
                      </td>
                      <td>
                        <span className="badge-info">{user.role}</span>
                      </td>
                      <td className="text-sm text-gray-500">{user.lastActive}</td>
                      <td>
                        <div className="flex items-center gap-2">
                          <div className={`w-2 h-2 rounded-full ${user.status === 'online' ? 'bg-emerald-500' : user.status === 'away' ? 'bg-amber-500' : 'bg-gray-400'}`} />
                          <span className="text-sm capitalize">{user.status}</span>
                        </div>
                      </td>
                      <td>
                        <div className="flex gap-2">
                          <button className="btn-ghost text-xs px-2 py-1">Edit</button>
                          <button className="btn-ghost text-xs px-2 py-1 text-red-600 hover:bg-red-50">Remove</button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card max-w-3xl">
            <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Role Permissions</h3></div>
            <div className="card-body">
              <div className="space-y-3">
                {['Super Admin', 'Manager', 'Finance Admin', 'Sales Manager', 'HR Manager', 'Viewer'].map((role) => (
                  <div key={role} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div>
                      <p className="text-sm font-medium text-gray-900">{role}</p>
                      <p className="text-xs text-gray-500">
                        {role === 'Super Admin' ? 'Full access to all modules' :
                          role === 'Viewer' ? 'Read-only access' :
                            `Manage ${role.replace(' Admin', '').replace(' Manager', '')} module`}
                      </p>
                    </div>
                    <button className="btn-ghost text-xs">Configure</button>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'notifications' && (
        <div className="max-w-3xl">
          <div className="card">
            <div className="card-header flex items-center gap-2">
              <Bell className="w-4 h-4 text-gray-500" />
              <h3 className="text-sm font-semibold text-gray-700">Notification Preferences</h3>
            </div>
            <div className="card-body space-y-4">
              {[
                { label: 'New Sales Orders', description: 'Get notified when new sales orders are created', enabled: true },
                { label: 'Low Stock Alerts', description: 'Alert when products fall below reorder level', enabled: true },
                { label: 'Invoice Overdue', description: 'Notify when invoices become overdue', enabled: true },
                { label: 'Leave Requests', description: 'HR notifications for new leave requests', enabled: false },
                { label: 'Purchase Order Updates', description: 'Status changes on purchase orders', enabled: true },
                { label: 'Project Milestones', description: 'Notify when project milestones are reached', enabled: false },
                { label: 'Payroll Processing', description: 'Alerts for payroll processing status', enabled: true },
                { label: 'System Maintenance', description: 'Scheduled system maintenance notifications', enabled: true },
              ].map((pref, i) => (
                <div key={i} className="flex items-center justify-between p-3 rounded-lg hover:bg-gray-50">
                  <div>
                    <p className="text-sm font-medium text-gray-900">{pref.label}</p>
                    <p className="text-xs text-gray-500">{pref.description}</p>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input type="checkbox" defaultChecked={pref.enabled} className="sr-only peer" />
                    <div className="w-9 h-5 bg-gray-300 rounded-full peer peer-checked:bg-blue-600 after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:after:translate-x-full" />
                  </label>
                </div>
              ))}
              <div className="flex justify-end pt-2">
                <button className="btn-primary"><Save className="w-4 h-4" /> Save Preferences</button>
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'system' && (
        <div className="max-w-3xl">
          <div className="card mb-6">
            <div className="card-header flex items-center gap-2">
              <Database className="w-4 h-4 text-gray-500" />
              <h3 className="text-sm font-semibold text-gray-700">System Information</h3>
            </div>
            <div className="card-body space-y-3">
              {[
                { label: 'Application Version', value: '2.4.1' },
                { label: 'Database', value: 'PostgreSQL 16.2' },
                { label: 'Last Backup', value: 'May 19, 2026, 3:00 AM CST' },
                { label: 'Storage Used', value: '42.5 GB / 100 GB (42.5%)' },
                { label: 'API Rate Limit', value: '1,000 requests/minute' },
                { label: 'Uptime', value: '99.97% (last 30 days)' },
              ].map((item, i) => (
                <div key={i} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
                  <span className="text-sm text-gray-600">{item.label}</span>
                  <span className="text-sm font-medium">{item.value}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Data Management</h3></div>
            <div className="card-body space-y-3">
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium">Backup Database</p>
                  <p className="text-xs text-gray-500">Create a full system backup</p>
                </div>
                <button className="btn-secondary text-sm">Run Backup</button>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium">Export All Data</p>
                  <p className="text-xs text-gray-500">Download all data as CSV/Excel</p>
                </div>
                <button className="btn-secondary text-sm">Export</button>
              </div>
              <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium">Clear Cache</p>
                  <p className="text-xs text-gray-500">Clear application cache and temporary data</p>
                </div>
                <button className="btn-secondary text-sm">Clear</button>
              </div>
              <div className="flex items-center justify-between p-3 bg-red-50 rounded-lg">
                <div>
                  <p className="text-sm font-medium text-red-800">Reset Demo Data</p>
                  <p className="text-xs text-red-600">Reset all data to demo defaults</p>
                </div>
                <button className="btn-danger text-sm">Reset</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
