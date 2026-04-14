// src/hooks/useDatabases.js
// Shared hook that fetches the database list and normalises each entry to
// { id, name }.  Eliminates the copy-pasted fetch pattern across 13 pages.

import { useState, useEffect } from 'react';
import { databaseApi } from '../lib/api';

/**
 * @returns {{
 *   databases: Array<{id: string, name: string}>,
 *   loading: boolean,
 *   error: string,
 *   refetch: () => void,
 * }}
 */
export function useDatabases() {
  const [databases, setDatabases] = useState([]);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState('');

  const fetch = async () => {
    setLoading(true);
    setError('');
    try {
      const response = await databaseApi.listDatabases();
      setDatabases(
        (response.databases || []).map(db => ({
          id:   db.databaseId,
          name: db.name,
        }))
      );
    } catch (err) {
      setError(`Error fetching databases: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetch(); }, []);   // eslint-disable-line react-hooks/exhaustive-deps

  return { databases, loading, error, refetch: fetch };
}
