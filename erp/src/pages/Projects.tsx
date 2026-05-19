import { useState } from 'react';
import { Plus, Download, FolderKanban, Clock, DollarSign, CheckCircle2 } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import PageHeader from '../components/PageHeader';
import Tabs from '../components/Tabs';
import StatusBadge from '../components/StatusBadge';
import KpiCard from '../components/KpiCard';
import { formatCurrency, formatDate } from '../components/FormatHelpers';
import { projects, tasks } from '../data/mockData';

const tasksByStatus = ['todo', 'in-progress', 'review', 'done'].map((s) => ({
  status: s.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase()),
  count: tasks.filter((t) => t.status === s).length,
}));

const PRIORITY_COLORS: Record<string, string> = {
  low: 'bg-gray-100 text-gray-700',
  medium: 'bg-blue-100 text-blue-700',
  high: 'bg-amber-100 text-amber-700',
  critical: 'bg-red-100 text-red-700',
};

const tabs = [
  { id: 'projects', label: 'Projects', count: projects.length },
  { id: 'tasks', label: 'Tasks', count: tasks.length },
  { id: 'board', label: 'Task Board' },
];

export default function Projects() {
  const [activeTab, setActiveTab] = useState('projects');

  const activeProjects = projects.filter((p) => p.status === 'active').length;
  const totalBudget = projects.reduce((s, p) => s + p.budget, 0);
  const totalSpent = projects.reduce((s, p) => s + p.spent, 0);
  const totalHoursLogged = tasks.reduce((s, t) => s + t.loggedHours, 0);

  return (
    <div>
      <PageHeader
        title="Projects"
        subtitle="Manage projects, tasks, and track progress."
        actions={
          <>
            <button className="btn-secondary"><Download className="w-4 h-4" /> Export</button>
            <button className="btn-primary"><Plus className="w-4 h-4" /> New Project</button>
          </>
        }
      />

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard title="Active Projects" value={String(activeProjects)} icon={FolderKanban} color="blue" />
        <KpiCard title="Total Budget" value={formatCurrency(totalBudget)} icon={DollarSign} color="emerald" />
        <KpiCard title="Total Spent" value={formatCurrency(totalSpent)} icon={DollarSign} color="amber" />
        <KpiCard title="Hours Logged" value={`${totalHoursLogged}h`} icon={Clock} color="purple" />
      </div>

      <Tabs tabs={tabs} activeTab={activeTab} onChange={setActiveTab} />

      {activeTab === 'projects' && (
        <div>
          <div className="card mb-6">
            <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">Tasks by Status</h3></div>
            <div className="card-body">
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={tasksByStatus}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="status" tick={{ fontSize: 12 }} />
                  <YAxis tick={{ fontSize: 12 }} allowDecimals={false} />
                  <Tooltip />
                  <Bar dataKey="count" fill="#2563eb" radius={[4, 4, 0, 0]} barSize={40} name="Tasks" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project) => (
              <div key={project.id} className="card p-5">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <h3 className="font-semibold text-gray-900">{project.name}</h3>
                    <p className="text-xs text-gray-500 mt-1">{project.client}</p>
                  </div>
                  <StatusBadge status={project.status} />
                </div>
                <p className="text-sm text-gray-600 mb-4 line-clamp-2">{project.description}</p>

                <div className="mb-4">
                  <div className="flex justify-between text-sm mb-1">
                    <span className="text-gray-500">Progress</span>
                    <span className="font-medium">{project.progress}%</span>
                  </div>
                  <div className="w-full h-2 bg-gray-200 rounded-full">
                    <div
                      className={`h-2 rounded-full transition-all ${project.progress === 100 ? 'bg-emerald-500' : project.progress > 50 ? 'bg-blue-500' : 'bg-amber-500'}`}
                      style={{ width: `${project.progress}%` }}
                    />
                  </div>
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Manager</span>
                    <span className="font-medium">{project.manager}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Budget</span>
                    <span className="font-medium">{formatCurrency(project.budget)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Spent</span>
                    <span className={`font-medium ${project.spent > project.budget * 0.9 ? 'text-red-600' : ''}`}>
                      {formatCurrency(project.spent)} ({Math.round((project.spent / project.budget) * 100)}%)
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Timeline</span>
                    <span className="font-medium text-xs">{formatDate(project.startDate)} - {formatDate(project.endDate)}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-500">Priority</span>
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${PRIORITY_COLORS[project.priority]}`}>
                      {project.priority.charAt(0).toUpperCase() + project.priority.slice(1)}
                    </span>
                  </div>
                </div>

                <div className="mt-4 pt-3 border-t border-gray-100">
                  <div className="flex items-center gap-1">
                    <span className="text-xs text-gray-500">Team:</span>
                    <div className="flex -space-x-2">
                      {project.teamMembers.slice(0, 4).map((member, i) => (
                        <div
                          key={i}
                          className="w-6 h-6 rounded-full bg-blue-100 border-2 border-white flex items-center justify-center text-[10px] font-medium text-blue-700"
                          title={member}
                        >
                          {member.split(' ').map((n) => n[0]).join('')}
                        </div>
                      ))}
                      {project.teamMembers.length > 4 && (
                        <div className="w-6 h-6 rounded-full bg-gray-100 border-2 border-white flex items-center justify-center text-[10px] font-medium text-gray-600">
                          +{project.teamMembers.length - 4}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {activeTab === 'tasks' && (
        <div className="card">
          <div className="card-header"><h3 className="text-sm font-semibold text-gray-700">All Tasks</h3></div>
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Project</th>
                  <th>Assignee</th>
                  <th>Priority</th>
                  <th>Due Date</th>
                  <th>Hours</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {tasks.map((task) => {
                  const project = projects.find((p) => p.id === task.projectId);
                  return (
                    <tr key={task.id}>
                      <td>
                        <div>
                          <p className="font-medium">{task.title}</p>
                          <p className="text-xs text-gray-500 line-clamp-1">{task.description}</p>
                        </div>
                      </td>
                      <td className="text-sm">{project?.name}</td>
                      <td>
                        <div className="flex items-center gap-2">
                          <div className="w-6 h-6 rounded-full bg-blue-100 flex items-center justify-center text-[10px] font-medium text-blue-700">
                            {task.assignee.split(' ').map((n) => n[0]).join('')}
                          </div>
                          <span className="text-sm">{task.assignee}</span>
                        </div>
                      </td>
                      <td>
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${PRIORITY_COLORS[task.priority]}`}>
                          {task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
                        </span>
                      </td>
                      <td>{formatDate(task.dueDate)}</td>
                      <td>
                        <span className="text-sm">{task.loggedHours}/{task.estimatedHours}h</span>
                      </td>
                      <td><StatusBadge status={task.status} /></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'board' && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {['todo', 'in-progress', 'review', 'done'].map((status) => (
            <div key={status} className="bg-gray-100 rounded-xl p-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-semibold text-gray-700">
                  {status.replace(/-/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase())}
                </h3>
                <span className="badge-gray">{tasks.filter((t) => t.status === status).length}</span>
              </div>
              <div className="space-y-3">
                {tasks.filter((t) => t.status === status).map((task) => {
                  const project = projects.find((p) => p.id === task.projectId);
                  return (
                    <div key={task.id} className="bg-white rounded-lg p-3 border border-gray-200 shadow-sm">
                      <p className="text-sm font-medium text-gray-900 mb-1">{task.title}</p>
                      <p className="text-xs text-gray-500 mb-2">{project?.name}</p>
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-1">
                          <div className="w-5 h-5 rounded-full bg-blue-100 flex items-center justify-center text-[9px] font-medium text-blue-700">
                            {task.assignee.split(' ').map((n) => n[0]).join('')}
                          </div>
                          <span className="text-xs text-gray-500">{task.assignee.split(' ')[0]}</span>
                        </div>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${PRIORITY_COLORS[task.priority]}`}>
                          {task.priority}
                        </span>
                      </div>
                      {task.status === 'done' && (
                        <div className="mt-2 flex items-center gap-1 text-emerald-600">
                          <CheckCircle2 className="w-3 h-3" />
                          <span className="text-xs">{task.loggedHours}h logged</span>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
