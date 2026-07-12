import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NC Law & Policy Search · P42",
  description:
    "Semantic search over public North Carolina child-welfare statutes and NCDHHS policy, with grounded, source-cited answers. A P42 demonstrator.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;450;500;600&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
