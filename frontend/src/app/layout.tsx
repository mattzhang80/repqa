import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "RepQA",
  description: "Smartphone-based rep quality analysis for shoulder rehab exercises",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-gray-50 text-gray-900">
        <nav className="bg-white border-b border-gray-200 px-6 py-3 flex items-center gap-6">
          <Link href="/" className="text-lg font-semibold text-brand-600 hover:text-brand-700">
            RepQA
          </Link>
          <Link href="/" className="text-sm text-gray-600 hover:text-gray-900">
            Sessions
          </Link>
          <Link href="/upload" className="text-sm text-gray-600 hover:text-gray-900">
            Upload
          </Link>
          <span className="ml-auto text-xs text-gray-400">
            Wall Slide &amp; Band ER Side
          </span>
        </nav>

        <div className="max-w-6xl mx-auto px-6 py-8">{children}</div>

        <div className="fixed bottom-0 inset-x-0 bg-amber-50 border-t border-amber-200 py-2 px-6 text-center text-xs text-amber-800">
          Stop if pain &gt;3/10 or if you feel instability/apprehension.
        </div>
      </body>
    </html>
  );
}
