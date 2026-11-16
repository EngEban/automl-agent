import { notFound } from "next/navigation";
import { DesignSystemPreview } from "@/components/shared/design-system-preview";

export const dynamic = "force-dynamic";

export default function DesignPreviewPage() {
  if (process.env.NODE_ENV !== "development") notFound();
  return <DesignSystemPreview />;
}
