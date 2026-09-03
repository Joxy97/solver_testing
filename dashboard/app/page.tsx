'use client';

import { ChangeEvent, useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  Binary,
  Boxes,
  CircleAlert,
  Cpu,
  FileJson,
  FolderOpen,
  Gauge,
  Orbit,
  RefreshCw,
  Trophy,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  NativeSelect,
  NativeSelectOption,
} from '@/components/ui/native-select';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

type CsvRow = Record<string, string>;
type ResultSet = { name: string; root: string; manifest: File };
type Batch = {
  name: string;
  root: string;
  files: File[];
  qubos: CsvRow[];
  resultSets: ResultSet[];
};
type QuboDetail = {
  seed?: number;
  qubo?: {
    num_variables?: number;
    linear?: number[][];
    quadratic?: number[][];
  };
  validation?: { warnings?: { code: string }[] };
};
type ResultDetail = {
  solution?: { energy?: number; sample?: number[] };
  timing?: { wall_seconds?: number };
  solver?: {
    name?: string;
    instance_name?: string;
    device?: string;
    version?: string;
  };
  verification?: { passed?: boolean };
};

const COLORS = [
  '#67e8f9',
  '#a3e635',
  '#c084fc',
  '#fb7185',
  '#fbbf24',
  '#60a5fa',
  '#34d399',
  '#f472b6',
];

function filePath(file: File) {
  return (
    (file as File & { webkitRelativePath?: string }).webkitRelativePath ||
    file.name
  ).replaceAll('\\', '/');
}

function parseCsv(text: string): CsvRow[] {
  const records: string[][] = [];
  let record: string[] = [];
  let value = '';
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else if (char === '"') quoted = false;
      else value += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') {
      record.push(value);
      value = '';
    } else if (char === '\n') {
      record.push(value.replace(/\r$/, ''));
      if (record.some(Boolean)) records.push(record);
      record = [];
      value = '';
    } else value += char;
  }
  if (value || record.length) {
    record.push(value.replace(/\r$/, ''));
    records.push(record);
  }
  const [headers = [], ...rows] = records;
  return rows.map((row) =>
    Object.fromEntries(
      headers.map((header, index) => [header, row[index] ?? '']),
    ),
  );
}

function numeric(value: string | undefined) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function formatNumber(value: number, digits = 3) {
  if (!Number.isFinite(value)) return '—';
  if (Math.abs(value) >= 1_000_000) return value.toExponential(2);
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: digits,
  }).format(value);
}

function formatBytes(bytes: number) {
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 10 ? 1 : 2)} ${units[unit]}`;
}

function solverLabel(row: CsvRow) {
  const name =
    (row.file || '')
      .split('/')
      .pop()
      ?.replace(/\.json$/i, '') || '';
  const marker = name.indexOf('--');
  return marker >= 0 ? name.slice(marker + 2) : row.solver_type || 'unknown';
}

function StatCard({
  label,
  value,
  note,
  icon: Icon,
}: {
  label: string;
  value: string;
  note: string;
  icon: typeof Activity;
}) {
  return (
    <Card className="metric-card">
      <CardContent className="flex items-start justify-between gap-4">
        <div>
          <p className="text-sm text-slate-400">{label}</p>
          <p className="mt-2 font-mono text-3xl font-semibold tracking-tight text-white">
            {value}
          </p>
          <p className="mt-2 text-xs text-slate-500">{note}</p>
        </div>
        <span className="grid size-9 place-items-center rounded-lg border border-cyan-300/15 bg-cyan-300/7 text-cyan-300">
          <Icon className="size-4" />
        </span>
      </CardContent>
    </Card>
  );
}

function ChartCard({
  title,
  note,
  children,
}: {
  title: string;
  note: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="panel">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <p className="text-xs text-slate-500">{note}</p>
      </CardHeader>
      <CardContent className="h-[310px]">{children}</CardContent>
    </Card>
  );
}

export default function Home() {
  const inputRef = useRef<HTMLInputElement>(null);
  const [batch, setBatch] = useState<Batch | null>(null);
  const [results, setResults] = useState<CsvRow[]>([]);
  const [resultSet, setResultSet] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [selectedProblem, setSelectedProblem] = useState('');
  const [problemDetail, setProblemDetail] = useState<QuboDetail | null>(null);
  const [selectedResult, setSelectedResult] = useState('');
  const [resultDetail, setResultDetail] = useState<ResultDetail | null>(null);
  const [solverFilter, setSolverFilter] = useState('all');

  async function activateResultSet(next: string, loadedBatch = batch) {
    if (!loadedBatch) return;
    const set = loadedBatch.resultSets.find((item) => item.name === next);
    setResultSet(next);
    setSelectedResult('');
    setResultDetail(null);
    setResults(set ? parseCsv(await set.manifest.text()) : []);
  }

  async function loadBatch(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files || []);
    event.target.value = '';
    if (!files.length) return;
    setLoading(true);
    setError('');
    try {
      const manifests = files.filter((file) =>
        /\/qubos\/manifest\.csv$/i.test(filePath(file)),
      );
      if (manifests.length !== 1) {
        throw new Error(
          `Select exactly one batch folder. Found ${manifests.length} qubos manifests.`,
        );
      }
      const manifestPath = filePath(manifests[0]);
      const root = manifestPath.slice(0, -'/qubos/manifest.csv'.length);
      const name = root.split('/').pop() || root;
      const resultSets = files
        .filter(
          (file) =>
            filePath(file).startsWith(`${root}/`) &&
            /\/results(?:\(\d+\))?\/manifest\.csv$/i.test(filePath(file)),
        )
        .map((manifest) => {
          const path = filePath(manifest);
          const resultRoot = path.slice(0, -'/manifest.csv'.length);
          return {
            name: resultRoot.split('/').pop() || 'results',
            root: resultRoot,
            manifest,
          };
        })
        .sort((a, b) =>
          a.name.localeCompare(b.name, undefined, { numeric: true }),
        );
      const nextBatch = {
        name,
        root,
        files,
        qubos: parseCsv(await manifests[0].text()),
        resultSets,
      };
      setBatch(nextBatch);
      setSelectedProblem(nextBatch.qubos[0]?.instance_id || '');
      setProblemDetail(null);
      setSolverFilter('all');
      if (resultSets.length)
        await activateResultSet(resultSets.at(-1)!.name, nextBatch);
      else {
        setResultSet('');
        setResults([]);
      }
    } catch (caught) {
      setBatch(null);
      setResults([]);
      setError(
        caught instanceof Error ? caught.message : 'Could not read this batch.',
      );
    } finally {
      setLoading(false);
    }
  }

  async function loadProblemDetail(instanceId: string) {
    if (!batch) return;
    setSelectedProblem(instanceId);
    setProblemDetail(null);
    const row = batch.qubos.find((item) => item.instance_id === instanceId);
    const fileName = row?.file || `${instanceId}.json`;
    const file = batch.files.find(
      (item) => filePath(item) === `${batch.root}/qubos/${fileName}`,
    );
    if (!file) return setError(`Could not find ${fileName}.`);
    if (
      file.size > 250 * 1024 * 1024 &&
      !window.confirm(
        `${file.name} is ${formatBytes(file.size)}. Loading it may make the browser unresponsive. Continue?`,
      )
    )
      return;
    setLoading(true);
    try {
      setProblemDetail(JSON.parse(await file.text()));
    } catch {
      setError(`Could not parse ${file.name}.`);
    } finally {
      setLoading(false);
    }
  }

  async function loadResultDetail(fileName: string) {
    if (!batch || !resultSet) return;
    setSelectedResult(fileName);
    setResultDetail(null);
    const set = batch.resultSets.find((item) => item.name === resultSet);
    const normalized = fileName.replaceAll('\\', '/');
    const file = batch.files.find(
      (item) =>
        filePath(item) === `${set?.root}/${normalized}` ||
        filePath(item).endsWith(`/${resultSet}/${normalized}`),
    );
    if (!file) return setError(`Could not find ${fileName}.`);
    setLoading(true);
    try {
      setResultDetail(JSON.parse(await file.text()));
    } catch {
      setError(`Could not parse ${file.name}.`);
    } finally {
      setLoading(false);
    }
  }

  const solvers = useMemo(
    () => [...new Set(results.map(solverLabel))].sort(),
    [results],
  );
  const visibleResults = useMemo(
    () =>
      results.filter(
        (row) => solverFilter === 'all' || solverLabel(row) === solverFilter,
      ),
    [results, solverFilter],
  );
  const energyData = useMemo(() => {
    const byProblem = new Map<string, Record<string, string | number>>();
    results.forEach((row) => {
      const current = byProblem.get(row.instance_id) || {
        instance: row.instance_id,
      };
      current[solverLabel(row)] = numeric(row.energy);
      byProblem.set(row.instance_id, current);
    });
    return [...byProblem.values()];
  }, [results]);
  const runtimeData = useMemo(
    () =>
      solvers.map((solver) => {
        const rows = results.filter((row) => solverLabel(row) === solver);
        return {
          solver,
          seconds:
            rows.reduce((sum, row) => sum + numeric(row.wall_seconds), 0) /
            Math.max(1, rows.length),
        };
      }),
    [results, solvers],
  );
  const leaderboard = useMemo(
    () =>
      solvers
        .map((solver) => {
          const rows = results.filter((row) => solverLabel(row) === solver);
          let wins = 0;
          let gap = 0;
          rows.forEach((row) => {
            const peers = results.filter(
              (peer) => peer.instance_id === row.instance_id,
            );
            const best = Math.min(...peers.map((peer) => numeric(peer.energy)));
            const delta = numeric(row.energy) - best;
            if (Math.abs(delta) < 1e-9) wins += 1;
            gap += delta;
          });
          return {
            solver,
            wins,
            avgGap: gap / Math.max(1, rows.length),
            avgTime:
              rows.reduce((sum, row) => sum + numeric(row.wall_seconds), 0) /
              Math.max(1, rows.length),
          };
        })
        .sort((a, b) => b.wins - a.wins || a.avgGap - b.avgGap),
    [results, solvers],
  );

  const histogram = useMemo(() => {
    if (!problemDetail?.qubo) return [];
    const coefficients = [
      ...(problemDetail.qubo.linear || []).map((item: number[]) =>
        Number(item[1]),
      ),
      ...(problemDetail.qubo.quadratic || []).map((item: number[]) =>
        Number(item[2]),
      ),
    ].filter(Number.isFinite);
    if (!coefficients.length) return [];
    const min = Math.min(...coefficients);
    const max = Math.max(...coefficients);
    const width = (max - min || 1) / 12;
    const bins = Array.from({ length: 12 }, (_, index) => ({
      range: `${formatNumber(min + index * width, 1)}`,
      count: 0,
    }));
    coefficients.forEach((value) => {
      bins[Math.min(11, Math.floor((value - min) / width))].count += 1;
    });
    return bins;
  }, [problemDetail]);

  const selectedProblemRow = batch?.qubos.find(
    (row) => row.instance_id === selectedProblem,
  );
  const ones =
    resultDetail?.solution?.sample?.filter((value: number) => value === 1)
      .length || 0;
  const bits = resultDetail?.solution?.sample?.length || 0;
  const bestEnergy = results.length
    ? Math.min(...results.map((row) => numeric(row.energy)))
    : 0;

  useEffect(() => {
    const context = (
      document as Document & {
        modelContext?: {
          registerTool: (
            tool: Record<string, unknown>,
            options?: { signal: AbortSignal },
          ) => void | Promise<void>;
        };
      }
    ).modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    void Promise.resolve(
      context.registerTool(
        {
          name: 'get_qubo_batch_summary',
          title: 'Get QUBO batch summary',
          description:
            'Read the summary of the single QUBO batch currently loaded in the dashboard.',
          inputSchema: {
            type: 'object',
            properties: {},
            additionalProperties: false,
          },
          annotations: { readOnlyHint: true, untrustedContentHint: true },
          execute: () =>
            batch
              ? {
                  batch: batch.name,
                  problems: batch.qubos.length,
                  active_result_set: resultSet || null,
                  solutions: results.length,
                  solver_instances: solvers,
                  best_energy: results.length ? bestEnergy : null,
                }
              : { batch: null, message: 'No batch is loaded.' },
        },
        { signal: lifecycle.signal },
      ),
    ).catch(() => undefined);
    return () => lifecycle.abort();
  }, [batch, resultSet, results.length, solvers, bestEnergy]);

  return (
    <main className="min-h-screen bg-background text-foreground">
      <input
        ref={inputRef}
        type="file"
        multiple
        className="hidden"
        onChange={loadBatch}
        onClick={(event) =>
          event.currentTarget.setAttribute('webkitdirectory', '')
        }
      />
      <header className="sticky top-0 z-30 border-b border-white/8 bg-[#090d14]/92 px-5 py-3 backdrop-blur-xl lg:px-8">
        <div className="mx-auto flex max-w-[1680px] flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <span className="grid size-10 place-items-center rounded-xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-300">
              <Orbit className="size-5" />
            </span>
            <div>
              <p className="font-mono text-[11px] uppercase tracking-[0.2em] text-cyan-300/75">
                QUBO Lab
              </p>
              <h1 className="text-lg font-semibold tracking-tight">
                Batch Inspector
              </h1>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {batch && (
              <div className="hidden text-right sm:block">
                <p className="text-sm font-medium">{batch.name}</p>
                <p className="text-xs text-slate-500">
                  local session · one batch
                </p>
              </div>
            )}
            <Button
              size="lg"
              onClick={() => inputRef.current?.click()}
              disabled={loading}
            >
              {loading ? (
                <RefreshCw className="animate-spin" />
              ) : (
                <FolderOpen />
              )}
              {batch ? 'Change batch' : 'Load batch folder'}
            </Button>
          </div>
        </div>
      </header>

      {!batch ? (
        <section className="mx-auto grid min-h-[calc(100vh-70px)] max-w-[1680px] place-items-center px-5 py-12">
          <div className="grid w-full max-w-5xl gap-5 lg:grid-cols-[1fr_300px]">
            <Card className="panel min-h-[520px]">
              <CardContent className="grid min-h-[470px] place-items-center">
                <div className="max-w-md text-center">
                  <div className="mx-auto mb-5 grid size-16 place-items-center rounded-2xl border border-dashed border-cyan-300/35 bg-cyan-300/5 text-cyan-300">
                    <FolderOpen className="size-7" />
                  </div>
                  <h2 className="text-3xl font-semibold tracking-tight">
                    Open one experiment batch
                  </h2>
                  <p className="mt-3 text-base leading-7 text-slate-400">
                    Choose a batch containing <code>qubos</code> and optional{' '}
                    <code>results</code> folders. Nothing leaves your browser.
                  </p>
                  <Button
                    className="mt-7"
                    size="lg"
                    onClick={() => inputRef.current?.click()}
                  >
                    <FolderOpen /> Choose batch
                  </Button>
                  {error && (
                    <p className="mt-5 text-sm text-rose-400">{error}</p>
                  )}
                </div>
              </CardContent>
            </Card>
            <div className="grid content-start gap-4">
              {[
                ['Problems', Boxes],
                ['Solution runs', Activity],
                ['Solver instances', Cpu],
              ].map(([label, Icon]) => (
                <StatCard
                  key={label as string}
                  label={label as string}
                  value="—"
                  note="Waiting for a batch"
                  icon={Icon as typeof Activity}
                />
              ))}
            </div>
          </div>
        </section>
      ) : (
        <section className="mx-auto max-w-[1680px] px-5 py-6 lg:px-8">
          {error && (
            <div className="mb-5 flex items-center gap-3 rounded-xl border border-rose-400/20 bg-rose-400/8 px-4 py-3 text-sm text-rose-300">
              <CircleAlert className="size-4" />
              {error}
              <button className="ml-auto" onClick={() => setError('')}>
                Dismiss
              </button>
            </div>
          )}
          <div className="mb-5 grid gap-4 sm:grid-cols-2 xl:grid-cols-5">
            <StatCard
              label="QUBOs"
              value={String(batch.qubos.length)}
              note={`${formatNumber(numeric(batch.qubos[0]?.num_variables), 0)} variables each`}
              icon={Boxes}
            />
            <StatCard
              label="Solutions"
              value={String(results.length)}
              note={`${batch.resultSets.length} result set${batch.resultSets.length === 1 ? '' : 's'}`}
              icon={Activity}
            />
            <StatCard
              label="Solvers"
              value={String(solvers.length)}
              note="named configurations"
              icon={Cpu}
            />
            <StatCard
              label="Best energy"
              value={results.length ? formatNumber(bestEnergy) : '—'}
              note="minimum in selected results"
              icon={Trophy}
            />
            <StatCard
              label="Quadratic terms"
              value={formatNumber(numeric(batch.qubos[0]?.quadratic_terms), 0)}
              note={`${formatNumber(numeric(batch.qubos[0]?.['parameter.quadratic_density']) * 100, 2)}% requested density`}
              icon={Binary}
            />
          </div>

          <div className="mb-5 flex flex-wrap items-center justify-between gap-3 rounded-xl border border-white/7 bg-[#101722] px-4 py-3">
            <div>
              <p className="text-xs uppercase tracking-[0.16em] text-slate-500">
                Active result set
              </p>
              <p className="mt-0.5 text-sm text-slate-300">
                Compare one solve run at a time
              </p>
            </div>
            <NativeSelect
              className="w-52"
              value={resultSet}
              onChange={(event) => activateResultSet(event.target.value)}
              disabled={!batch.resultSets.length}
            >
              {!batch.resultSets.length && (
                <NativeSelectOption value="">
                  No results found
                </NativeSelectOption>
              )}
              {batch.resultSets.map((item) => (
                <NativeSelectOption key={item.name} value={item.name}>
                  {item.name}
                </NativeSelectOption>
              ))}
            </NativeSelect>
          </div>

          <Tabs defaultValue="overview" className="gap-5">
            <TabsList variant="line" className="border-b border-white/8 px-1">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="problems">Problems</TabsTrigger>
              <TabsTrigger value="results">Results</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-5">
              {results.length ? (
                <>
                  <div className="grid gap-5 xl:grid-cols-2">
                    <ChartCard
                      title="Energy by QUBO"
                      note="Lower is better. Each line is a named solver instance."
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart
                          data={energyData}
                          margin={{ top: 10, right: 12, left: 6, bottom: 10 }}
                        >
                          <CartesianGrid stroke="#ffffff0d" vertical={false} />
                          <XAxis
                            dataKey="instance"
                            stroke="#64748b"
                            tick={{ fontSize: 11 }}
                          />
                          <YAxis
                            stroke="#64748b"
                            tick={{ fontSize: 11 }}
                            width={72}
                          />
                          <Tooltip
                            contentStyle={{
                              background: '#111827',
                              border: '1px solid #ffffff18',
                              borderRadius: 10,
                            }}
                          />
                          <Legend />
                          {solvers.map((solver, index) => (
                            <Line
                              key={solver}
                              type="monotone"
                              dataKey={solver}
                              stroke={COLORS[index % COLORS.length]}
                              strokeWidth={2}
                              dot={{ r: 3 }}
                              connectNulls
                            />
                          ))}
                        </LineChart>
                      </ResponsiveContainer>
                    </ChartCard>
                    <ChartCard
                      title="Average runtime"
                      note="Mean wall-clock seconds for each solver instance."
                    >
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={runtimeData}
                          margin={{ top: 10, right: 12, left: 6, bottom: 40 }}
                        >
                          <CartesianGrid stroke="#ffffff0d" vertical={false} />
                          <XAxis
                            dataKey="solver"
                            stroke="#64748b"
                            tick={{ fontSize: 11 }}
                            angle={-18}
                            textAnchor="end"
                          />
                          <YAxis stroke="#64748b" tick={{ fontSize: 11 }} />
                          <Tooltip
                            contentStyle={{
                              background: '#111827',
                              border: '1px solid #ffffff18',
                              borderRadius: 10,
                            }}
                          />
                          <Bar
                            dataKey="seconds"
                            fill="#a3e635"
                            radius={[6, 6, 0, 0]}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </ChartCard>
                  </div>
                  <Card className="panel">
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Trophy className="size-4 text-lime-300" /> Solver
                        leaderboard
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Rank</TableHead>
                            <TableHead>Solver instance</TableHead>
                            <TableHead className="text-right">
                              Best-energy wins
                            </TableHead>
                            <TableHead className="text-right">
                              Average gap
                            </TableHead>
                            <TableHead className="text-right">
                              Average seconds
                            </TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {leaderboard.map((row, index) => (
                            <TableRow key={row.solver}>
                              <TableCell className="font-mono text-slate-500">
                                {String(index + 1).padStart(2, '0')}
                              </TableCell>
                              <TableCell className="font-medium text-cyan-200">
                                {row.solver}
                              </TableCell>
                              <TableCell className="text-right">
                                {row.wins}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {formatNumber(row.avgGap)}
                              </TableCell>
                              <TableCell className="text-right font-mono">
                                {formatNumber(row.avgTime)}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                </>
              ) : (
                <Card className="panel">
                  <CardContent className="grid min-h-64 place-items-center text-center">
                    <div>
                      <Gauge className="mx-auto size-8 text-slate-600" />
                      <h3 className="mt-3 text-lg font-medium">
                        No solution results yet
                      </h3>
                      <p className="mt-1 text-slate-500">
                        This batch contains QUBOs but no results manifest.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              )}
            </TabsContent>

            <TabsContent
              value="problems"
              className="grid gap-5 xl:grid-cols-[minmax(0,1.15fr)_minmax(420px,.85fr)]"
            >
              <Card className="panel">
                <CardHeader>
                  <CardTitle>Problem instances</CardTitle>
                </CardHeader>
                <CardContent>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Instance</TableHead>
                        <TableHead className="text-right">Variables</TableHead>
                        <TableHead className="text-right">Linear</TableHead>
                        <TableHead className="text-right">Quadratic</TableHead>
                        <TableHead>Status</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {batch.qubos.map((row) => (
                        <TableRow
                          key={row.instance_id}
                          className="cursor-pointer"
                          data-state={
                            selectedProblem === row.instance_id
                              ? 'selected'
                              : undefined
                          }
                          onClick={() => loadProblemDetail(row.instance_id)}
                        >
                          <TableCell className="font-mono text-cyan-200">
                            {row.instance_id}
                          </TableCell>
                          <TableCell className="text-right font-mono">
                            {formatNumber(numeric(row.num_variables), 0)}
                          </TableCell>
                          <TableCell className="text-right font-mono">
                            {formatNumber(numeric(row.linear_terms), 0)}
                          </TableCell>
                          <TableCell className="text-right font-mono">
                            {formatNumber(numeric(row.quadratic_terms), 0)}
                          </TableCell>
                          <TableCell>
                            <Badge
                              variant={
                                row.validation_status === 'valid'
                                  ? 'secondary'
                                  : 'outline'
                              }
                            >
                              {row.validation_status || 'unknown'}
                            </Badge>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </CardContent>
              </Card>
              <Card className="panel">
                <CardHeader>
                  <CardTitle>{selectedProblem || 'Problem detail'}</CardTitle>
                  <p className="text-xs text-slate-500">
                    {selectedProblemRow
                      ? `${selectedProblemRow.file} · select a row to load coefficients`
                      : 'Select an instance'}
                  </p>
                </CardHeader>
                <CardContent>
                  {problemDetail ? (
                    <div className="space-y-5">
                      <div className="grid grid-cols-2 gap-3">
                        {[
                          ['Seed', problemDetail.seed],
                          ['Variables', problemDetail.qubo?.num_variables],
                          ['Linear terms', problemDetail.qubo?.linear?.length],
                          [
                            'Quadratic terms',
                            problemDetail.qubo?.quadratic?.length,
                          ],
                        ].map(([label, value]) => (
                          <div key={label as string} className="detail-cell">
                            <p>{label}</p>
                            <strong>{formatNumber(Number(value), 0)}</strong>
                          </div>
                        ))}
                      </div>
                      <div>
                        <h4 className="mb-3 text-sm font-medium">
                          Coefficient distribution
                        </h4>
                        <div className="h-56">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart data={histogram}>
                              <CartesianGrid
                                stroke="#ffffff0d"
                                vertical={false}
                              />
                              <XAxis
                                dataKey="range"
                                tick={{ fontSize: 10 }}
                                stroke="#64748b"
                              />
                              <YAxis tick={{ fontSize: 10 }} stroke="#64748b" />
                              <Tooltip
                                contentStyle={{
                                  background: '#111827',
                                  border: '1px solid #ffffff18',
                                  borderRadius: 10,
                                }}
                              />
                              <Bar
                                dataKey="count"
                                fill="#67e8f9"
                                radius={[4, 4, 0, 0]}
                              />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                      <div className="rounded-lg border border-white/7 bg-black/15 p-3 text-xs text-slate-400">
                        <p className="font-medium text-slate-300">
                          Validation warnings
                        </p>
                        <p className="mt-1">
                          {problemDetail.validation?.warnings
                            ?.map((warning: { code: string }) => warning.code)
                            .join(', ') || 'None'}
                        </p>
                      </div>
                    </div>
                  ) : (
                    <div className="grid min-h-72 place-items-center text-center">
                      <div>
                        <FileJson className="mx-auto size-8 text-slate-600" />
                        <p className="mt-3 text-slate-400">
                          Select a problem row to inspect its coefficients.
                        </p>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="results" className="space-y-5">
              <div className="flex justify-end">
                <NativeSelect
                  className="w-60"
                  value={solverFilter}
                  onChange={(event) => setSolverFilter(event.target.value)}
                >
                  <NativeSelectOption value="all">
                    All solver instances
                  </NativeSelectOption>
                  {solvers.map((solver) => (
                    <NativeSelectOption key={solver} value={solver}>
                      {solver}
                    </NativeSelectOption>
                  ))}
                </NativeSelect>
              </div>
              <div className="grid gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(390px,.75fr)]">
                <Card className="panel">
                  <CardHeader>
                    <CardTitle>Solution records</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>QUBO</TableHead>
                          <TableHead>Solver instance</TableHead>
                          <TableHead>Device</TableHead>
                          <TableHead className="text-right">Energy</TableHead>
                          <TableHead className="text-right">Seconds</TableHead>
                          <TableHead>Status</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {visibleResults.map((row) => (
                          <TableRow
                            key={`${row.instance_id}-${row.file}`}
                            className="cursor-pointer"
                            data-state={
                              selectedResult === row.file
                                ? 'selected'
                                : undefined
                            }
                            onClick={() => loadResultDetail(row.file)}
                          >
                            <TableCell className="font-mono text-cyan-200">
                              {row.instance_id}
                            </TableCell>
                            <TableCell>{solverLabel(row)}</TableCell>
                            <TableCell className="max-w-40 truncate text-slate-400">
                              {row.device}
                            </TableCell>
                            <TableCell className="text-right font-mono">
                              {formatNumber(numeric(row.energy))}
                            </TableCell>
                            <TableCell className="text-right font-mono">
                              {formatNumber(numeric(row.wall_seconds))}
                            </TableCell>
                            <TableCell>
                              <Badge
                                variant={
                                  row.status === 'completed' ||
                                  row.status === 'optimal'
                                    ? 'secondary'
                                    : 'destructive'
                                }
                              >
                                {row.status}
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
                <Card className="panel">
                  <CardHeader>
                    <CardTitle>Solution detail</CardTitle>
                    <p className="truncate text-xs text-slate-500">
                      {selectedResult || 'Select a result row'}
                    </p>
                  </CardHeader>
                  <CardContent>
                    {resultDetail ? (
                      <div className="space-y-5">
                        <div className="grid grid-cols-2 gap-3">
                          <div className="detail-cell">
                            <p>Verified energy</p>
                            <strong>
                              {formatNumber(resultDetail.solution?.energy ?? Number.NaN)}
                            </strong>
                          </div>
                          <div className="detail-cell">
                            <p>Wall time</p>
                            <strong>
                              {formatNumber(resultDetail.timing?.wall_seconds ?? Number.NaN)}{' '}
                              s
                            </strong>
                          </div>
                          <div className="detail-cell">
                            <p>Ones</p>
                            <strong>{formatNumber(ones, 0)}</strong>
                          </div>
                          <div className="detail-cell">
                            <p>Zeros</p>
                            <strong>{formatNumber(bits - ones, 0)}</strong>
                          </div>
                        </div>
                        <div>
                          <div className="mb-2 flex justify-between text-xs text-slate-400">
                            <span>Binary composition</span>
                            <span>
                              {bits ? formatNumber((ones / bits) * 100, 1) : 0}%
                              ones
                            </span>
                          </div>
                          <div className="flex h-4 overflow-hidden rounded-full bg-slate-800">
                            <span
                              className="bg-cyan-300"
                              style={{
                                width: `${bits ? (ones / bits) * 100 : 0}%`,
                              }}
                            />
                          </div>
                        </div>
                        <dl className="space-y-3 text-sm">
                          {[
                            ['Solver', resultDetail.solver?.name],
                            ['Instance', resultDetail.solver?.instance_name],
                            ['Device', resultDetail.solver?.device],
                            ['Version', resultDetail.solver?.version],
                            [
                              'Verification',
                              resultDetail.verification?.passed
                                ? 'Passed'
                                : 'Failed',
                            ],
                          ].map(([label, value]) => (
                            <div
                              key={label}
                              className="flex justify-between gap-4 border-b border-white/6 pb-2"
                            >
                              <dt className="text-slate-500">{label}</dt>
                              <dd className="text-right text-slate-200">
                                {String(value ?? '—')}
                              </dd>
                            </div>
                          ))}
                        </dl>
                      </div>
                    ) : (
                      <div className="grid min-h-72 place-items-center text-center">
                        <div>
                          <Activity className="mx-auto size-8 text-slate-600" />
                          <p className="mt-3 text-slate-400">
                            Select a result row for its sample and metadata.
                          </p>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </section>
      )}
    </main>
  );
}
