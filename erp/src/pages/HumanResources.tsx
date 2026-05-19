import { useState } from 'react';
import { Plus, Download, Mail, Phone, Search } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from 'recharts';
import PageHeader from '../components/PageHeader';
import Tabs from '../components/Tabs';
import StatusBadge from '../components/StatusBadge';
import { formatCurrency, formatDate } from '../components/FormatHelpers';
import { employees, departments, payrollRecords, leaveRequests } from '../data/mockData';

const DEPT_COLORS = ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#db2777', '#ea580c'];

const tabs = [
  { id: 'employees', label: 'Employees', count: employees.length },
  { id: 'departments', label: 'Departments', count: departments.length },
  { id: 'payroll', label: 'Payroll', count: payrollRecords.length },
  { id: 'leave', label: 'Leave Requests', count: leaveRequests.length },
];

export default function HumanResources() {
  const [activeTab, setActiveTab] = useState('employees');
  const [search, setSearch] = useState('');

  const filteredEmployees = employees.filter(
    (e) =>
      e.name.toLowerCase().includes(search.toLowerCase()) ||
      e.department.toLowerCase().includes(search.toLowerCase()) ||
      e.position.toLowerCase().includes(search.toLowerCase())
  );

  const deptData = departments.map((d) => ({ name: d.name, count: d.employeeCount }));
  const statusBreakdown = [
    { name: 'Active', value: employees.filter((e) => e.status === 'active').length },
    { name: 'On Leave', value: employees.filter((e) => e.status === 'on-leave').length },
    { name: 'Terminated', value: employees.filter((e) => e.status === 'terminated').length },
  ];

  return (
    <div>
      <PageHeader
        title="Human Resources"
        subtitle="Manage employees, departments, payroll, and leave requests."
        actions={
          <>
            <button className="btn-secondary"><Download className="w-4 h-4" /> Export</button>
            <button className="btn-primary"><Plus className="w-4 h-4" /> Add Employee</button>
          </>
        }
      />

      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === 'employees' && (
        <div>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <div className="card lg:col-span-2">
              <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Employees by Department</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={deptData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                    <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-20} textAnchor="end" height={60} />
                    <YAxis tick={{ fontSize: 12 }} />
                    <Tooltip />
                    <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} barSize={32} name="Employees" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            <div className="card">
              <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Status Breakdown</h3></div>
              <div className="card-body">
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={statusBreakdown} cx="50%" cy="50%" innerRadius={50} outerRadius={80} paddingAngle={3} dataKey="value" nameKey="name">
                      <Cell fill="#059669" />
                      <Cell fill="#d97706" />
                      <Cell fill="#dc2626" />
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
              <h3 className="text-sm font-semibold text-gray-700">Employee Directory</h3>
              <div className="relative">
                <Search className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search employees..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="input pl-9 w-64"
                />
              </div>
            </div>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>Employee</th>
                    <th>Department</th>
                    <th>Position</th>
                    <th>Contact</th>
                    <th>Hire Date</th>
                    <th>Salary</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEmployees.map((emp) => (
                    <tr key={emp.id}>
                      <td>
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 rounded-full bg-blue-100 flex items-center justify-center text-sm font-medium text-blue-700">
                            {emp.name.split(' ').map((n) => n[0]).join('')}
                          </div>
                          <div>
                            <p className="font-medium text-gray-900">{emp.name}</p>
                            <p className="text-xs text-gray-500">{emp.id}</p>
                          </div>
                        </div>
                      </td>
                      <td>{emp.department}</td>
                      <td>{emp.position}</td>
                      <td>
                        <div className="space-y-1">
                          <div className="flex items-center gap-1 text-xs text-gray-500"><Mail className="w-3 h-3" />{emp.email}</div>
                          <div className="flex items-center gap-1 text-xs text-gray-500"><Phone className="w-3 h-3" />{emp.phone}</div>
                        </div>
                      </td>
                      <td className="text-gray-600">{formatDate(emp.hireDate)}</td>
                      <td className="font-medium">{formatCurrency(emp.salary)}</td>
                      <td><StatusBadge status={emp.status} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'departments' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {departments.map((dept, idx) => (
            <div key={dept.id} className="card p-5">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-lg flex items-center justify-center" style={{ backgroundColor: `${DEPT_COLORS[idx % DEPT_COLORS.length]}15`, color: DEPT_COLORS[idx % DEPT_COLORS.length] }}>
                  <span className="text-lg font-bold">{dept.name[0]}</span>
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{dept.name}</h3>
                  <p className="text-xs text-gray-500">Head: {dept.head}</p>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Employees</span>
                  <span className="font-medium">{dept.employeeCount}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Budget</span>
                  <span className="font-medium">{formatCurrency(dept.budget)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {activeTab === 'payroll' && (
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Payroll Records</h3></div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Employee</th>
                  <th>Department</th>
                  <th>Period</th>
                  <th>Base Salary</th>
                  <th>Bonus</th>
                  <th>Deductions</th>
                  <th>Net Pay</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {payrollRecords.map((pr) => (
                  <tr key={pr.id}>
                    <td className="font-medium">{pr.employeeName}</td>
                    <td>{pr.department}</td>
                    <td>{pr.period}</td>
                    <td>{formatCurrency(pr.baseSalary)}</td>
                    <td className="text-emerald-600">+{formatCurrency(pr.bonus)}</td>
                    <td className="text-red-600">-{formatCurrency(pr.deductions)}</td>
                    <td className="font-bold">{formatCurrency(pr.netPay)}</td>
                    <td><StatusBadge status={pr.status} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'leave' && (
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Leave Requests</h3></div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Employee</th>
                  <th>Type</th>
                  <th>Start Date</th>
                  <th>End Date</th>
                  <th>Reason</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {leaveRequests.map((lr) => (
                  <tr key={lr.id}>
                    <td className="font-medium">{lr.employeeName}</td>
                    <td className="capitalize">{lr.type}</td>
                    <td>{formatDate(lr.startDate)}</td>
                    <td>{formatDate(lr.endDate)}</td>
                    <td className="text-gray-600">{lr.reason}</td>
                    <td><StatusBadge status={lr.status} /></td>
                    <td>
                      {lr.status === 'pending' && (
                        <div className="flex gap-2">
                          <button className="btn-success text-xs px-2 py-1">Approve</button>
                          <button className="btn-danger text-xs px-2 py-1">Reject</button>
                        </div>
                      )}
                    </td>
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
