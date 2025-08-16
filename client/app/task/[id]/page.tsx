"use client";

import { useState, useEffect } from "react";
import { useRouter, useParams } from "next/navigation";
import { 
  FileText, 
  Loader2, 
  CheckCircle, 
  AlertCircle,
  Clock,
  ArrowLeft,
  RefreshCw
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import api from "../../../lib/api";

type TaskStatus = {
  task_id: string;
  celery_task_id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress_percentage: number;
  progress_message: string;
  file_name: string;
  query: string;
  category: string;
  method: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
  error_message?: string;
  result?: {
    submitted_file_id: number;
    answer: string;
    accuracy_score: number;
    extracted_fields: Record<string, any>;
    retrieval_method: string;
    processing_time: number;
    evaluation: {
      overall_score: number;
      semantic_similarity: number;
      context_relevance: number;
    };
    document_info: {
      filename: string;
      text_length: number;
      extraction_method: string;
      pages: number;
    };
  };
  celery_status?: {
    state: string;
    info: any;
  };
};

export default function TaskStatusPage() {
  const router = useRouter();
  const params = useParams();
  const taskId = params.id as string;
  
  const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [refreshing, setRefreshing] = useState(false);

  const fetchTaskStatus = async (showRefreshLoader = false) => {
    try {
      if (showRefreshLoader) setRefreshing(true);
      
      const response = await api.get<TaskStatus>(`/v1/task/${taskId}/`);
      setTaskStatus(response.data);
      setError("");
      
      // Stop polling if task is completed or failed
      if (response.data.status === 'completed' || response.data.status === 'failed') {
        return false; // Stop polling
      }
      
      return true; // Continue polling
    } catch (err: any) {
      const errorMessage = err.response?.data?.error || "Failed to fetch task status";
      setError(errorMessage);
      return false; // Stop polling on error
    } finally {
      setLoading(false);
      if (showRefreshLoader) setRefreshing(false);
    }
  };

  useEffect(() => {
    if (!taskId) return;

    // Initial fetch
    fetchTaskStatus();

    // Set up polling every 2 seconds
    const pollInterval = setInterval(async () => {
      const shouldContinue = await fetchTaskStatus();
      if (!shouldContinue) {
        clearInterval(pollInterval);
      }
    }, 2000);

    // Cleanup on unmount
    return () => clearInterval(pollInterval);
  }, [taskId]);

  const getStatusIcon = () => {
    if (!taskStatus) return <Loader2 className="h-6 w-6 animate-spin text-blue-500" />;
    
    switch (taskStatus.status) {
      case 'pending':
        return <Clock className="h-6 w-6 text-yellow-500" />;
      case 'processing':
        return <Loader2 className="h-6 w-6 animate-spin text-blue-500" />;
      case 'completed':
        return <CheckCircle className="h-6 w-6 text-green-500" />;
      case 'failed':
        return <AlertCircle className="h-6 w-6 text-red-500" />;
      default:
        return <Clock className="h-6 w-6 text-gray-500" />;
    }
  };

  const getStatusColor = () => {
    if (!taskStatus) return "text-gray-500";
    
    switch (taskStatus.status) {
      case 'pending':
        return "text-yellow-600";
      case 'processing':
        return "text-blue-600";
      case 'completed':
        return "text-green-600";
      case 'failed':
        return "text-red-600";
      default:
        return "text-gray-600";
    }
  };

  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const handleRefresh = () => {
    fetchTaskStatus(true);
  };

  const handleBackToDashboard = () => {
    router.push('/dashboard');
  };

  if (loading && !taskStatus) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-4" />
          <p className="text-gray-600">Loading task status...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <FileText className="h-8 w-8 text-blue-600" />
              <h1 className="ml-2 text-xl font-semibold text-gray-900">
                Document Processing Status
              </h1>
            </div>
            <div className="flex items-center gap-2">
              <Button
                onClick={handleRefresh}
                variant="outline"
                size="sm"
                disabled={refreshing}
                className="flex items-center gap-2"
              >
                <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Button
                onClick={handleBackToDashboard}
                variant="outline"
                size="sm"
                className="flex items-center gap-2"
              >
                <ArrowLeft className="h-4 w-4" />
                Back to Dashboard
              </Button>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="space-y-6">
          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Task Overview */}
          {taskStatus && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-3">
                  {getStatusIcon()}
                  <span className={`capitalize ${getStatusColor()}`}>
                    {taskStatus.status}
                  </span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-sm font-medium text-gray-700">Progress</span>
                    <span className="text-sm text-gray-500">{taskStatus.progress_percentage}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${taskStatus.progress_percentage}%` }}
                    ></div>
                  </div>
                  {taskStatus.progress_message && (
                    <p className="text-sm text-gray-600">{taskStatus.progress_message}</p>
                  )}
                </div>

                {/* Task Details */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-4 border-t">
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-1">File Name</h4>
                    <p className="text-sm text-gray-900">{taskStatus.file_name}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-1">Category</h4>
                    <p className="text-sm text-gray-900">{taskStatus.category}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-1">Method</h4>
                    <p className="text-sm text-gray-900 capitalize">{taskStatus.method}</p>
                  </div>
                  <div>
                    <h4 className="text-sm font-medium text-gray-500 mb-1">Task ID</h4>
                    <p className="text-sm text-gray-900 font-mono">{taskStatus.task_id}</p>
                  </div>
                </div>

                {/* Query */}
                <div className="pt-4 border-t">
                  <h4 className="text-sm font-medium text-gray-500 mb-2">Query</h4>
                  <p className="text-sm text-gray-900 bg-gray-50 p-3 rounded">{taskStatus.query}</p>
                </div>

                {/* Timestamps */}
                <div className="pt-4 border-t">
                  <h4 className="text-sm font-medium text-gray-500 mb-2">Timeline</h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Created:</span>
                      <span className="text-gray-900">{formatTimestamp(taskStatus.created_at)}</span>
                    </div>
                    {taskStatus.started_at && (
                      <div className="flex justify-between">
                        <span className="text-gray-600">Started:</span>
                        <span className="text-gray-900">{formatTimestamp(taskStatus.started_at)}</span>
                      </div>
                    )}
                    {taskStatus.completed_at && (
                      <div className="flex justify-between">
                        <span className="text-gray-600">Completed:</span>
                        <span className="text-gray-900">{formatTimestamp(taskStatus.completed_at)}</span>
                      </div>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Error Message */}
          {taskStatus?.status === 'failed' && taskStatus.error_message && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-red-600">
                  <AlertCircle className="h-5 w-5" />
                  Error Details
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-red-700 bg-red-50 p-4 rounded">
                  {taskStatus.error_message}
                </p>
              </CardContent>
            </Card>
          )}

          {/* Results */}
          {taskStatus?.status === 'completed' && taskStatus.result && (
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-green-600">
                  <CheckCircle className="h-5 w-5" />
                  Processing Results
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Answer */}
                <div>
                  <h4 className="text-sm font-medium text-gray-500 mb-2">AI Response</h4>
                  <div className="bg-blue-50 p-4 rounded-lg">
                    {(() => {
                      try {
                        const parsedAnswer = JSON.parse(taskStatus.result.answer);
                        return (
                          <div className="space-y-3">
                            {Object.entries(parsedAnswer).map(([key, value]) => (
                              <div key={key} className="flex flex-col sm:flex-row sm:items-start gap-2">
                                <span className="text-sm font-medium text-blue-700 min-w-[120px] capitalize">
                                  {key.replace(/_/g, ' ')}:
                                </span>
                                <span className="text-sm text-gray-800 flex-1">
                                  {value === null ? (
                                    <em className="text-gray-500">Not specified</em>
                                  ) : (
                                    String(value)
                                  )}
                                </span>
                              </div>
                            ))}
                          </div>
                        );
                      } catch (error) {
                        return (
                          <p className="text-sm leading-relaxed whitespace-pre-wrap">
                            {taskStatus.result.answer}
                          </p>
                        );
                      }
                    })()}
                  </div>
                </div>

                {/* Accuracy Score */}
                <div>
                  <h4 className="text-sm font-medium text-gray-500 mb-2">Accuracy Score</h4>
                  <div className="bg-green-50 p-4 rounded-lg">
                    <div className="flex items-center justify-between">
                      <span className="text-2xl font-bold text-green-700">
                        {taskStatus.result.accuracy_score?.toFixed(1) || '0.0'}%
                      </span>
                      <span className="text-sm text-gray-600">
                        Processing Time: {taskStatus.result.processing_time?.toFixed(2)}s
                      </span>
                    </div>
                  </div>
                </div>

                {/* Document Info */}
                {taskStatus.result.document_info && (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t">
                    <div className="text-center">
                      <p className="text-lg font-semibold text-gray-900">
                        {taskStatus.result.document_info.pages || 0}
                      </p>
                      <p className="text-xs text-gray-500">Pages</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-semibold text-gray-900">
                        {taskStatus.result.document_info.text_length || 0}
                      </p>
                      <p className="text-xs text-gray-500">Characters</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-semibold text-gray-900">
                        {taskStatus.result.retrieval_method || 'N/A'}
                      </p>
                      <p className="text-xs text-gray-500">Method</p>
                    </div>
                    <div className="text-center">
                      <p className="text-lg font-semibold text-gray-900">
                        {taskStatus.result.evaluation?.overall_score ? 
                          (taskStatus.result.evaluation.overall_score * 100).toFixed(0) + '%' : 'N/A'}
                      </p>
                      <p className="text-xs text-gray-500">Confidence</p>
                    </div>
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}