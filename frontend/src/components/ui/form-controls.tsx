"use client";

import { useId, useState } from "react";
import type {
  InputHTMLAttributes,
  SelectHTMLAttributes,
} from "react";
import { Eye, EyeOff, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface FieldProps {
  label: string;
  hint?: string;
  error?: string;
  id?: string;
  className?: string;
}

function FieldMessages({ id, hint, error }: { id: string; hint?: string; error?: string }) {
  return (
    <>
      {hint && !error && <span className="field-hint" id={`${id}-hint`}>{hint}</span>}
      {error && <span className="field-error" id={`${id}-error`} role="alert">{error}</span>}
    </>
  );
}

export interface TextInputProps extends FieldProps, Omit<InputHTMLAttributes<HTMLInputElement>, "id"> {}

export function TextInput({ label, hint, error, id: providedId, className, ...props }: TextInputProps) {
  const generatedId = useId();
  const id = providedId ?? generatedId;
  const describedBy = error ? `${id}-error` : hint ? `${id}-hint` : undefined;

  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>{label}</label>
      <input
        className={cn("input", error && "input-error", className)}
        id={id}
        aria-invalid={Boolean(error)}
        aria-describedby={describedBy}
        {...props}
      />
      <FieldMessages id={id} hint={hint} error={error} />
    </div>
  );
}

export interface PasswordInputProps extends FieldProps, Omit<InputHTMLAttributes<HTMLInputElement>, "id" | "type"> {}

export function PasswordInput({ label, hint, error, id: providedId, className, ...props }: PasswordInputProps) {
  const generatedId = useId();
  const id = providedId ?? generatedId;
  const [visible, setVisible] = useState(false);
  const describedBy = error ? `${id}-error` : hint ? `${id}-hint` : undefined;

  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>{label}</label>
      <div className="input-with-action">
        <input
          className={cn("input", error && "input-error", className)}
          id={id}
          type={visible ? "text" : "password"}
          aria-invalid={Boolean(error)}
          aria-describedby={describedBy}
          {...props}
        />
        <button
          className="input-action"
          type="button"
          aria-label={visible ? "Hide password" : "Show password"}
          aria-pressed={visible}
          onClick={() => setVisible((value) => !value)}
        >
          {visible ? <EyeOff size={17} aria-hidden="true" /> : <Eye size={17} aria-hidden="true" />}
        </button>
      </div>
      <FieldMessages id={id} hint={hint} error={error} />
    </div>
  );
}

export interface SelectFieldProps extends FieldProps, Omit<SelectHTMLAttributes<HTMLSelectElement>, "id"> {
  options: Array<{ label: string; value: string }>;
}

export function SelectField({ label, hint, error, id: providedId, className, options, ...props }: SelectFieldProps) {
  const generatedId = useId();
  const id = providedId ?? generatedId;
  const describedBy = error ? `${id}-error` : hint ? `${id}-hint` : undefined;

  return (
    <div className="field">
      <label className="field-label" htmlFor={id}>{label}</label>
      <div className="select-wrap">
        <select
          className={cn("input select-input", error && "input-error", className)}
          id={id}
          aria-invalid={Boolean(error)}
          aria-describedby={describedBy}
          {...props}
        >
          {options.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
        </select>
        <ChevronDown size={16} aria-hidden="true" />
      </div>
      <FieldMessages id={id} hint={hint} error={error} />
    </div>
  );
}

export interface CheckboxFieldProps extends FieldProps, Omit<InputHTMLAttributes<HTMLInputElement>, "id" | "type"> {}

export function CheckboxField({ label, hint, error, id: providedId, className, ...props }: CheckboxFieldProps) {
  const generatedId = useId();
  const id = providedId ?? generatedId;
  const describedBy = error ? `${id}-error` : hint ? `${id}-hint` : undefined;

  return (
    <div className="field">
      <label className="checkbox-field" htmlFor={id}>
        <input className={cn("checkbox", className)} id={id} type="checkbox" aria-invalid={Boolean(error)} aria-describedby={describedBy} {...props} />
        <span>{label}</span>
      </label>
      <FieldMessages id={id} hint={hint} error={error} />
    </div>
  );
}

export interface ToggleProps {
  label: string;
  description?: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  disabled?: boolean;
}

export function Toggle({ label, description, checked, onCheckedChange, disabled = false }: ToggleProps) {
  const id = useId();
  return (
    <div className={cn("toggle-field", disabled && "is-disabled")}>
      <div className="toggle-copy">
        <span className="field-label" id={`${id}-label`}>{label}</span>
        {description && <span className="field-hint" id={`${id}-description`}>{description}</span>}
      </div>
      <button
        className={cn("switch", checked && "switch-on")}
        type="button"
        role="switch"
        aria-checked={checked}
        aria-labelledby={`${id}-label`}
        aria-describedby={description ? `${id}-description` : undefined}
        disabled={disabled}
        onClick={() => onCheckedChange(!checked)}
      >
        <span className="switch-thumb" />
      </button>
    </div>
  );
}
