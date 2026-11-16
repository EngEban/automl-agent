"use client";

import { useRef, useState } from "react";
import type { ChangeEvent, FormEvent } from "react";
import { ApiError, ApiTimeoutError } from "@/lib/api/client";
import { Button } from "@/components/ui/button";
import { Alert } from "@/components/ui/feedback";
import { Modal } from "@/components/ui/modal";
import { TextInput } from "@/components/ui/form-controls";
import { useAuth } from "@/features/auth/auth-provider";
import { uploadDatasetSession } from "@/features/dashboard/dashboard-api";
import type { SessionDetails } from "@/types/api";

const SUPPORTED_EXTENSIONS = new Set(["csv", "tsv", "json", "jsonl", "xlsx", "xls"]);

export function UploadDatasetDialog({
  open,
  token,
  onClose,
  onCreated,
}: {
  open: boolean;
  token: string;
  onClose: () => void;
  onCreated: (session: SessionDetails) => void;
}) {
  const { logout } = useAuth();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState("");
  const [nameEdited, setNameEdited] = useState(false);
  const [targetColumn, setTargetColumn] = useState("");
  const [fileError, setFileError] = useState("");
  const [nameError, setNameError] = useState("");
  const [requestError, setRequestError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const submitLock = useRef(false);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.currentTarget.files?.[0] ?? null;
    setFile(selected);
    setFileError("");
    if (selected && !nameEdited) setName(selected.name.replace(/\.[^.]+$/, "").slice(0, 255));
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (submitLock.current) return;
    setRequestError("");

    const extension = file?.name.split(".").pop()?.toLowerCase();
    if (!file) {
      setFileError("Choose a dataset file to upload.");
      return;
    }
    if (!extension || !SUPPORTED_EXTENSIONS.has(extension)) {
      setFileError("Choose a CSV, TSV, JSON, JSONL, XLSX, or XLS file.");
      return;
    }
    if (!name.trim()) {
      setNameError("Enter a name for this dataset session.");
      return;
    }

    submitLock.current = true;
    setSubmitting(true);
    try {
      const created = await uploadDatasetSession(token, {
        name: name.trim(),
        file,
        targetColumn,
      });
      onCreated(created);
      onClose();
    } catch (error) {
      if (error instanceof ApiTimeoutError) {
        setRequestError("The upload took too long. Check your connection and try again.");
      } else if (error instanceof ApiError && error.status === 413) {
        setRequestError("This file exceeds the server's upload size limit.");
      } else if (error instanceof ApiError && (error.status === 400 || error.status === 422)) {
        setRequestError("The dataset could not be processed. Check the file and session details.");
      } else if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
        setRequestError("Your session has expired. Sign in again and retry the upload.");
        logout();
      } else if (error instanceof TypeError) {
        setRequestError("We could not reach the service. Check your connection and try again.");
      } else {
        setRequestError("The dataset could not be uploaded. Please try again.");
      }
    } finally {
      submitLock.current = false;
      setSubmitting(false);
    }
  }

  return (
    <Modal open={open} onClose={onClose} title="Start a new analysis">
      <div className="upload-dialog-content">
        <p>Upload a dataset to create a session. Starting and monitoring an analysis will be available in the AI workspace.</p>
        {requestError && <Alert tone="error" title="Upload failed">{requestError}</Alert>}
        <form className="upload-dataset-form" onSubmit={handleSubmit} noValidate>
          <TextInput
            label="Dataset file"
            type="file"
            accept=".csv,.tsv,.json,.jsonl,.xlsx,.xls"
            onChange={handleFileChange}
            error={fileError || undefined}
            required
          />
          <TextInput
            label="Dataset name"
            value={name}
            maxLength={255}
            onChange={(event) => {
              setName(event.target.value);
              setNameEdited(true);
              setNameError("");
            }}
            placeholder="e.g. Customer churn"
            error={nameError || undefined}
            required
          />
          <TextInput
            label="Target column (optional)"
            value={targetColumn}
            onChange={(event) => setTargetColumn(event.target.value)}
            placeholder="e.g. churned"
          />
          <div className="upload-dialog-actions">
            <Button type="button" variant="ghost" disabled={submitting} onClick={onClose}>Cancel</Button>
            <Button type="submit" loading={submitting}>Create dataset session</Button>
          </div>
        </form>
      </div>
    </Modal>
  );
}
