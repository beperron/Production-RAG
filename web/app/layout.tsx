import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NC Law & Policy Search",
  description: "Semantic search over public North Carolina child-welfare statutes and NCDHHS policy — every result source-traceable.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
