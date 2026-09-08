import React, { useState } from 'react';
import { assessTransaction } from './services/api';
import ReactMarkdown from 'react-markdown';
import { ShieldAlert, ShieldCheck, AlertTriangle, Activity, Cpu } from 'lucide-react';

export default function App() {
  const [formData, setFormData] = useState({
    step: 1,
    amount: 500000.0,
    oldbalanceOrg: 500000.0,
    newbalanceOrig: 0.0,
    oldbalanceDest: 0.0,
    newbalanceDest: 0.0,
    is_transfer: 'TRANSFER'
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'number' ? parseFloat(value) || 0 : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const data = await assessTransaction(formData);
      setResult(data);
    } catch (err) {
      setError('Failed to reach FastAPI backend. Verify server is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  const getDecisionBadge = (decision) => {
    switch (decision) {
      case 'BLOCK TRANSACTION':
        return {
          bg: 'bg-red-950/80 border-red-500 text-red-400',
          icon: <ShieldAlert className="w-6 h-6 text-red-400" />,
          label: 'BLOCK TRANSACTION'
        };
      case 'FLAG FOR MANUAL REVIEW':
        return {
          bg: 'bg-amber-950/80 border-amber-500 text-amber-400',
          icon: <AlertTriangle className="w-6 h-6 text-amber-400" />,
          label: 'FLAG FOR MANUAL REVIEW'
        };
      default:
        return {
          bg: 'bg-emerald-950/80 border-emerald-500 text-emerald-400',
          icon: <ShieldCheck className="w-6 h-6 text-emerald-400" />,
          label: 'ALLOW TRANSACTION'
        };
    }
  };

  return (
    <div className="min-h-screen p-8 bg-slate-900 text-slate-100">
      <header className="max-w-7xl mx-auto mb-8 border-b border-slate-800 pb-4 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
            <Activity className="w-7 h-7 text-indigo-400" />
            Transaction Risk Scoring & Decision System
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            XGBoost ML Pipeline + SHAP XAI Engine + Gemini Compliance Auditing
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-slate-800 border border-slate-700 text-xs text-slate-300">
          <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
          FastAPI Engine Online
        </div>
      </header>

      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Transaction Input Form */}
        <section className="lg:col-span-5 bg-slate-800/50 border border-slate-700 rounded-xl p-6 shadow-xl backdrop-blur-sm">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-400" />
            Simulate Payment Telemetry
          </h2>

          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label className="block text-xs font-medium text-slate-400 mb-1">Transaction Type</label>
              <select
                name="is_transfer"
                value={formData.is_transfer}
                onChange={handleChange}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
              >
                <option value="TRANSFER">TRANSFER</option>
                <option value="CASH_OUT">CASH_OUT</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Step (Hour)</label>
                <input
                  type="number"
                  name="step"
                  value={formData.step}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Amount ($)</label>
                <input
                  type="number"
                  name="amount"
                  value={formData.amount}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Origin Old Balance ($)</label>
                <input
                  type="number"
                  name="oldbalanceOrg"
                  value={formData.oldbalanceOrg}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Origin New Balance ($)</label>
                <input
                  type="number"
                  name="newbalanceOrig"
                  value={formData.newbalanceOrig}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Dest Old Balance ($)</label>
                <input
                  type="number"
                  name="oldbalanceDest"
                  value={formData.oldbalanceDest}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-slate-400 mb-1">Dest New Balance ($)</label>
                <input
                  type="number"
                  name="newbalanceDest"
                  value={formData.newbalanceDest}
                  onChange={handleChange}
                  className="w-full bg-slate-900 border border-slate-700 rounded-lg p-2.5 text-slate-200 focus:outline-none focus:border-indigo-500"
                />
              </div>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full mt-6 bg-indigo-600 hover:bg-indigo-500 text-white font-semibold py-3 px-4 rounded-lg transition duration-200 disabled:opacity-50"
            >
              {loading ? 'Evaluating Risk Telemetry...' : 'Assess Risk & Evaluate Policy'}
            </button>
          </form>

          {error && (
            <div className="mt-4 p-3 bg-red-950/50 border border-red-800 rounded-lg text-red-400 text-xs">
              {error}
            </div>
          )}
        </section>

        {/* Results & LLM Audit Output Panel */}
        <section className="lg:col-span-7 space-y-6">
          {result ? (
            <>
              {/* Decision Banner */}
              {(() => {
                const badge = getDecisionBadge(result.decision);
                return (
                  <div className={`p-6 border rounded-xl shadow-lg flex items-start gap-4 ${badge.bg}`}>
                    {badge.icon}
                    <div className="flex-1">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-xs font-semibold uppercase tracking-wider text-slate-300">
                          System Decision
                        </span>
                        <span className="text-xs font-mono font-bold px-2 py-1 bg-slate-900/60 rounded text-slate-200">
                          Risk Score: {(result.risk_score * 100).toFixed(2)}%
                        </span>
                      </div>
                      <h3 className="text-xl font-bold mb-2">{badge.label}</h3>
                      <p className="text-xs text-slate-300">
                        <strong className="text-slate-100">Primary Risk Driver:</strong> {result.primary_driver}
                      </p>
                    </div>
                  </div>
                );
              })()}

              {/* Gemini LLM Executive Audit Report */}
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6 shadow-xl">
                <h3 className="text-base font-semibold text-white mb-3 border-b border-slate-700 pb-2">
                   Gemini Compliance & AML Executive Audit
                </h3>
                <div className="prose prose-invert max-w-none text-slate-300 text-sm leading-relaxed">
                  <ReactMarkdown>{result.audit_summary}</ReactMarkdown>
                </div>
              </div>

              {/* Feature Vector Table */}
              <div className="bg-slate-800/50 border border-slate-700 rounded-xl p-6 shadow-xl">
                <h3 className="text-base font-semibold text-white mb-3 border-b border-slate-700 pb-2">
                  Calculated Feature Engineering Vector
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-xs">
                  {Object.entries(result.feature_vector).map(([key, value]) => (
                    <div key={key} className="bg-slate-900/70 p-2.5 rounded-lg border border-slate-800">
                      <span className="text-slate-400 block font-mono text-[11px]">{key}</span>
                      <span className="text-white font-semibold">{value.toString()}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          ) : (
            <div className="h-full bg-slate-800/30 border border-slate-800 border-dashed rounded-xl p-12 flex flex-col items-center justify-center text-center">
              <Activity className="w-12 h-12 text-slate-600 mb-3" />
              <h3 className="text-base font-medium text-slate-300">No Assessment Data</h3>
              <p className="text-xs text-slate-500 max-w-sm mt-1">
                Configure transaction parameters on the left and submit to view full ML inference, SHAP drivers, and Gemini compliance audits.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}