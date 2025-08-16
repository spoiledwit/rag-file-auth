"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { 
  Upload, 
  FileText, 
  Send, 
  Loader2, 
  LogOut, 
  MessageSquare,
  CheckCircle,
  AlertCircle
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import api from "../../lib/api";

type TaskResponse = {
  success: boolean;
  task_id: string;
  celery_task_id: string;
  message: string;
  status: string;
  file_name: string;
  query: string;
  category: string;
  method: string;
};

type Category = {
  id: number;
  category_name: string;
  prompt: string;
  description: string;
};

export default function DashboardPage() {
  const [file, setFile] = useState<File | null>(null);
  const [categories, setCategories] = useState<Category[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<Category | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const router = useRouter();

  // Fetch categories on component mount
  useEffect(() => {
    const fetchCategories = async () => {
      try {
        const response = await api.get<Category[]>("/v1/categories/");
        setCategories(response.data);
        if (response.data.length > 0) {
          setSelectedCategory(response.data[0]); // Select first category by default
        }
      } catch (err) {
        console.error("Failed to fetch categories:", err);
        setError("Failed to load categories");
      }
    };

    fetchCategories();
  }, []);

  const handleLogout = () => {
    localStorage.removeItem("token");
    router.push("/login");
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0];
      if (isValidFileType(droppedFile)) {
        setFile(droppedFile);
        setError("");
      } else {
        setError("Please upload a PDF, DOCX, or image file");
      }
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (isValidFileType(selectedFile)) {
        setFile(selectedFile);
        setError("");
      } else {
        setError("Please upload a PDF, DOCX, or image file");
      }
    }
  };

  const isValidFileType = (file: File) => {
    const allowedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "application/msword",
      "image/jpeg",
      "image/png",
      "image/jpg"
    ];
    return allowedTypes.includes(file.type);
  };

  const handleSubmit = async () => {
    if (!file || !selectedCategory) {
      setError("Please select a file and category");
      return;
    }

    if (!selectedCategory.prompt || selectedCategory.prompt.trim() === "") {
      setError("Selected category has no prompt configured. Please contact admin.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);
      formData.append("query", selectedCategory.prompt.trim());
      formData.append("category", selectedCategory.category_name);
      formData.append("method", "hybrid"); // Default to hybrid retrieval

      const response = await api.post<TaskResponse>("/v1/query-document/", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      // Redirect to task status page
      router.push(`/task/${response.data.task_id}`);
    } catch (err: any) {
      setError(
        err.response?.data?.error || "Failed to process your request. Please try again."
      );
      setLoading(false);
    }
  };

  const resetForm = () => {
    setFile(null);
    setSelectedCategory(categories.length > 0 ? categories[0] : null);
    setError("");
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <FileText className="h-8 w-8 text-blue-600" />
              <h1 className="ml-2 text-xl font-semibold text-gray-900">
                Document AI Assistant
              </h1>
            </div>
            <Button
              onClick={handleLogout}
              variant="outline"
              size="sm"
              className="flex items-center gap-2"
            >
              <LogOut className="h-4 w-4" />
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="space-y-8">
          {/* Upload Section */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Upload className="h-5 w-5" />
                Upload Document
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div
                className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                  dragActive
                    ? "border-blue-500 bg-blue-50"
                    : "border-gray-300 hover:border-gray-400"
                }`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
              >
                {file ? (
                  <div className="space-y-2">
                    <CheckCircle className="h-12 w-12 text-green-500 mx-auto" />
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                    <Button
                      onClick={() => setFile(null)}
                      variant="outline"
                      size="sm"
                      className="mt-2"
                    >
                      Remove
                    </Button>
                  </div>
                ) : (
                  <div className="space-y-4">
                    <Upload className="h-12 w-12 text-gray-400 mx-auto" />
                    <div>
                      <p className="text-sm font-medium text-gray-900">
                        Drop your document here, or{" "}
                        <label className="text-blue-600 hover:text-blue-500 cursor-pointer">
                          browse
                          <input
                            type="file"
                            className="hidden"
                            accept=".pdf,.doc,.docx,.jpg,.jpeg,.png"
                            onChange={handleFileSelect}
                          />
                        </label>
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Supports PDF, DOCX, and image files
                      </p>
                    </div>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Category Selection */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <MessageSquare className="h-5 w-5" />
                Select Analysis Category
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Category
                </label>
                <select
                  value={selectedCategory?.id || ""}
                  onChange={(e) => {
                    const categoryId = parseInt(e.target.value);
                    const category = categories.find(c => c.id === categoryId);
                    setSelectedCategory(category || null);
                  }}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  disabled={categories.length === 0}
                >
                  {categories.length === 0 ? (
                    <option>Loading categories...</option>
                  ) : (
                    categories.map((category) => (
                      <option key={category.id} value={category.id}>
                        {category.category_name}
                      </option>
                    ))
                  )}
                </select>
                {selectedCategory?.description && (
                  <p className="text-sm text-gray-500 mt-2">
                    {selectedCategory.description}
                  </p>
                )}
              </div>
              <div className="flex gap-2">
                <Button
                  onClick={handleSubmit}
                  disabled={!file || !selectedCategory || loading}
                  className="flex items-center gap-2"
                >
                  {loading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                  {loading ? "Processing..." : "Analyze Document"}
                </Button>
                {file && (
                  <Button onClick={resetForm} variant="outline">
                    Clear All
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Error Display */}
          {error && (
            <Alert variant="destructive">
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

        </div>
      </main>
    </div>
  );
}
