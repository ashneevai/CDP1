import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import App from "./App";

// Mock Query Client Provider and dynamic API calls
vi.mock("@tanstack/react-query", () => {
  return {
    useQuery: ({ queryKey }: { queryKey: string[] }) => {
      const key = queryKey[0];
      if (key === "review-tasks") {
        return {
          data: [
            {
              task_id: "live-task-1",
              claim_id: "clm-live-1",
              field_name: "billing_provider_npi",
              status: "OPEN",
              created_at: "2026-08-26T12:00:00Z",
              version: 1,
              assigned_to: null
            }
          ],
          isLoading: false,
          isError: false
        };
      }
      if (key === "review-task") {
        return {
          data: {
            task_id: "live-task-1",
            claim_id: "clm-live-1",
            field_name: "billing_provider_npi",
            status: "OPEN",
            created_at: "2026-08-26T12:00:00Z",
            version: 1,
            assigned_to: null,
            document_id: "doc-1",
            page_number: 1,
            crop_signed_url: null,
            ocr_candidates: ["1234567893"],
            vlm_candidate: "1234567893",
            validation_errors: ["Luhn check fail"],
            review_reason_codes: ["LOW_CONFIDENCE"],
            candidate_evidence: [],
            reference_evidence: [],
            system_recommendation: "1234567893",
            evidence_versions: {}
          },
          isLoading: false,
          isError: false
        };
      }
      if (key === "review-task-audit") {
        return {
          data: [
            {
              occurred_at: "2026-08-26T12:01:00Z",
              actor: "System Pipeline",
              event_type: "DOCUMENT_RECEIVED",
              task_version: 1,
              reason_code: "SYSTEM_INTAKE"
            }
          ],
          isLoading: false,
          isError: false
        };
      }
      return { data: [], isLoading: false, isError: false };
    },
    useMutation: () => ({ mutate: vi.fn(), isPending: false, isError: false, error: null }),
    useQueryClient: () => ({ invalidateQueries: vi.fn() })
  };
});

describe("Claims IDP Enterprise UI Tests", () => {
  it("renders side-by-side claim reviewer and tabs correctly", async () => {
    render(<App />);

    // 1. Sidebar is rendered and has the main branding
    expect(screen.getByText("Claims IDP")).toBeInTheDocument();
    expect(screen.getByText("Aarati Joshi")).toBeInTheDocument();

    // 2. Main tabs are loaded and present
    expect(screen.getByRole("tab", { name: /Dashboard/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Work Queue/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Document Review/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Analytics/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Audit Trail/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Settings/i })).toBeInTheDocument();

    // 3. Operational KPIs on Dashboard
    expect(screen.getByText("True STP")).toBeInTheDocument();
    expect(screen.getByText("Queue Documents")).toBeInTheDocument();
    expect(screen.getByText("Open Review Tasks")).toBeInTheDocument();

    // 4. Verify tab switches and headings
    fireEvent.click(screen.getByRole("tab", { name: /Work Queue/i }));
    expect(screen.getByText("Universal Healthcare Claims Queue")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Document Review/i }));
    expect(screen.getByText("Live Review Context")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Audit Trail/i }));
    expect(screen.getByText("Chronological Claims Pipeline Log")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("tab", { name: /Settings/i }));
    expect(screen.getByText("Confidence Thresholds & Active Business Rules")).toBeInTheDocument();
  });
});
